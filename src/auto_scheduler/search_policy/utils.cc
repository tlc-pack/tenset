/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/search_policy/utils.cc
 * \brief Common utilities
 */

#include "utils.h"

#include <tvm/tir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/support/parallel_for.h>

#include <algorithm>


namespace tvm {
// import the function from driver_api.cc
void GetBinds(const Array<te::Tensor>& args, bool compact,
              const std::unordered_map<te::Tensor, tir::Buffer>& binds,
              Map<te::Tensor, tir::Buffer>* out_binds, Array<ObjectRef>* out_arg_list);
}  // namespace tvm

namespace tvm {
namespace auto_scheduler {

Array<Integer> GetSpatialSplitStepIds(const State& s, int stage_id) {
  const auto& stage = s->stages[stage_id];
  const auto& pop = s->stages[stage_id]->op.as<te::ComputeOpNode>();
  ICHECK(pop != nullptr);
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  size_t reduce_count = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint)) {
      reduce_count++;
    }
  }

  Array<Integer> spatial_split_step_ids;
  for (int i = s->transform_steps.size() - 1; i >= 0; --i) {
    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    } else if (auto ps = s->transform_steps[i].as<SplitStepNode>()) {
      if (stage_id == ps->stage_id) {
        // Assume SplitStep on reduction axes are always after SplitStep on spatial axes.
        if (reduce_count) {
          reduce_count--;
        } else {
          spatial_split_step_ids.push_back(i);
        }
      }
    }
  }

  return spatial_split_step_ids;
}

std::vector<std::pair<int, int>> GetComputeLocationCandidates(const SearchTask& task,
                                                              const State& state, int stage_id) {
  int target_stage_id = GetSingleConsumerId(task, state, stage_id);
  if (target_stage_id < 0) {
    return {};
  }
  const Stage& target_stage = state->stages[target_stage_id];

  std::vector<std::pair<int, int>> candidates;
  bool target_compute_at_other = target_stage->compute_at == ComputeAtKind::kIter;
  bool target_is_tiled = IsTiled(target_stage);

  bool visited_reduce = false;
  // Enumerate compute_at location at target_stage
  // TODO(merrymercy): More analysis here to make smarter choices
  for (size_t i = 0; i < target_stage->iters.size(); ++i) {
    const Iterator& target_iter = target_stage->iters[i];
    if (target_iter->iter_kind == IteratorKind::kReduction) {
      visited_reduce = true;
      if (!target_is_tiled) {  // Do not go into reduce iter
        break;
      }
    } else if (target_iter->iter_kind == IteratorKind::kSpatial) {
      if (visited_reduce) {  // Do not go into inner tile
        break;
      }
    }

    if (target_iter->annotation == IteratorAnnotation::kUnroll) {
      // Do not go into the unroll region of const tensor indices
      break;
    }

    if (GetExtent(target_iter) == 1) {
      // Skip iterators with length of 1
      continue;
    }
    if (target_compute_at_other && target_iter->iter_kind == IteratorKind::kSpatial &&
        StrEndsWith(target_iter->name, ".0")) {
      // Skip the first level iterators if target stage compute_at another stage
      // In this case, the lengths of first level iterators are always one
      continue;
    }
    candidates.emplace_back(target_stage_id, i);

    if (state->attach_map->iter_to_attached_stages.count(std::make_pair(target_stage_id, i))) {
      break;
    }
  }

  // if the target_stage is already compute_at another stage X, try also compute_at X
  // We call stage X as `target_target_stage`
  if (target_compute_at_other) {
    int target_target_stage_id;
    target_target_stage_id = state->attach_map->stage_to_attach_iter.at(target_stage_id).first;
    const Stage& target_target_stage = state->stages[target_target_stage_id];

    for (size_t i = 0; i < target_target_stage->iters.size(); ++i) {
      const Iterator& target_target_iter = target_target_stage->iters[i];
      if (target_target_iter->iter_kind == IteratorKind::kReduction ||
          state->attach_map->iter_to_attached_stages.count(
              std::make_pair(target_target_stage_id, i))) {
        break;
      }

      if (target_target_iter->annotation == IteratorAnnotation::kUnroll) {
        // Do not go into the unroll region of const tensor indices
        break;
      }

      if (GetExtent(target_target_iter) == 1) {  // skip iterators with length of 1
        continue;
      }

      candidates.emplace_back(target_target_stage_id, i);
    }
  }

  return candidates;
}

State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids) {
  // Temporal object to be used if the input pointer is nullptr
  std::vector<int> temp_split_step_ids;
  if (spatial_split_step_ids == nullptr) {
    spatial_split_step_ids = &temp_split_step_ids;
  }
  std::vector<std::vector<Iterator>> space_levels;
  std::vector<std::vector<Iterator>> reduce_levels;
  std::vector<Iterator> space_outer, space_inner, reduce_outer, reduce_inner;
  Array<Iterator> split_res;

  for (const auto c : format) {
    if (tolower(c) == 's') {
      space_levels.emplace_back();
    } else if (tolower(c) == 'r') {
      reduce_levels.emplace_back();
    } else {
      LOG(FATAL) << "Invalid multi-level tiling format: " << format;
    }
  }
  size_t n_space = space_levels.size();
  size_t n_reduce = reduce_levels.size();

  spatial_split_step_ids->clear();

  State tmp_s = state;
  const Stage& stage = state->stages[stage_id];
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();

  for (const auto& iter : state->stages[stage_id]->iters) {
    if (!no_split_at_inner_name_set.count(iter->name)) {
      if (iter->iter_kind == IteratorKind::kSpatial) {
        ICHECK_GE(n_space, 1);

        if (n_space == 1) {
          space_levels[0].push_back(iter);
        } else {
          split_res = tmp_s.split(stage_id, iter, Array<Optional<Integer>>(n_space - 1, NullOpt));
          for (size_t i = 0; i < n_space; i++) {
            space_levels[i].push_back(split_res[i]);
          }
          spatial_split_step_ids->push_back(tmp_s->transform_steps.size() - 1);
        }
      } else if (iter->iter_kind == IteratorKind::kReduction) {
        ICHECK_GE(n_reduce, 1);

        if (n_reduce == 1) {
          reduce_levels[0].push_back(iter);
        } else {
          split_res = tmp_s.split(stage_id, iter, Array<Optional<Integer>>(n_reduce - 1, NullOpt));
          for (size_t i = 0; i < n_reduce; i++) {
            reduce_levels[i].push_back(split_res[i]);
          }
        }
      } else {
        LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
      }
    } else {
      if (iter->iter_kind == IteratorKind::kSpatial) {
        space_inner.push_back(iter);
      } else if (iter->iter_kind == IteratorKind::kReduction) {
        reduce_inner.push_back(iter);
      } else {
        LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
      }
    }
  }

  if (!space_outer.empty()) {
    ICHECK(!space_levels.empty());
    space_levels.front().insert(space_levels.front().begin(),
                                std::make_move_iterator(space_outer.begin()),
                                std::make_move_iterator(space_outer.end()));
  }
  if (!space_inner.empty()) {
    ICHECK(!space_levels.empty());
    space_levels.back().insert(space_levels.back().begin(),
                               std::make_move_iterator(space_inner.begin()),
                               std::make_move_iterator(space_inner.end()));
  }

  if (!reduce_outer.empty()) {
    ICHECK(!reduce_levels.empty());
    reduce_levels.front().insert(reduce_levels.front().begin(),
                                 std::make_move_iterator(reduce_outer.begin()),
                                 std::make_move_iterator(reduce_outer.end()));
  }
  if (!reduce_inner.empty()) {
    ICHECK(!reduce_levels.empty());
    reduce_levels.back().insert(reduce_levels.back().begin(),
                                std::make_move_iterator(reduce_inner.begin()),
                                std::make_move_iterator(reduce_inner.end()));
  }

  Array<Iterator> order;
  int space_ct = 0, reduce_ct = 0;
  for (const auto c : format) {
    if (tolower(c) == 's') {
      order.insert(order.end(), std::make_move_iterator(space_levels[space_ct].begin()),
                   std::make_move_iterator(space_levels[space_ct].end()));
      space_ct++;
    } else if (tolower(c) == 'r') {
      order.insert(order.end(), std::make_move_iterator(reduce_levels[reduce_ct].begin()),
                   std::make_move_iterator(reduce_levels[reduce_ct].end()));
      reduce_ct++;
    } else {
      LOG(FATAL) << "Invalid multi level tiling format: " << format;
    }
  }

  tmp_s.reorder(stage_id, order);
  return tmp_s;
}

State FollowTiling(const State& state, int stage_id, const std::vector<int>& split_step_ids,
                   int n_split) {
  if (n_split < 1 || n_split > 3) {
    LOG(FATAL) << "Invalid split parts, currently only support 1, 2 and 3";
  }
  // Apply up to three-level tiling structure:  space_L0, space_L1, space_L2
  std::vector<Iterator> space_0, space_1, space_2, space_3, tmp_order;
  Array<Iterator> split_res;

  auto pop = state->stages[stage_id]->op.as<te::ComputeOpNode>();
  ICHECK(pop != nullptr);
  const Stage& stage = state->stages[stage_id];
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  int no_split_at_inner_name_in_stage_cnt = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    no_split_at_inner_name_in_stage_cnt += no_split_at_inner_name_set.count(iter->name);
  }

  ICHECK_EQ(state->stages[stage_id]->iters.size() - no_split_at_inner_name_in_stage_cnt,
            split_step_ids.size());

  State tmp_s = state;
  int ct = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_kind == IteratorKind::kSpatial) {
      // For spatial iterator, split it into multi iterators
      if (!no_split_at_inner_name_set.count(iter->name)) {
        IteratorAnnotation ann_type = iter->annotation;
        split_res = tmp_s.follow_split(stage_id, iter, split_step_ids[ct], n_split);
        // Restore annotation. Move unroll and vectorize to inner, move parallel
        // to outer
        switch (ann_type) {
          case IteratorAnnotation::kUnroll:
            split_res.Set(n_split, tmp_s.unroll(stage_id, split_res[n_split]));
            break;
          case IteratorAnnotation::kVectorize:
            split_res.Set(n_split, tmp_s.vectorize(stage_id, split_res[n_split]));
            break;
          case IteratorAnnotation::kParallel:
            split_res.Set(0, tmp_s.parallel(stage_id, split_res[0]));
            break;
          default:
            break;
        }

        space_0.push_back(split_res[0]);
        space_1.push_back(split_res[1]);
        if (n_split >= 2) {
          space_2.push_back(split_res[2]);
          if (n_split == 3) {
            space_3.push_back(split_res[3]);
          }
        }
        ct++;
      } else {
        if (no_split_at_inner_name_set.count(iter->name)) {
          if (n_split == 1) {
            space_1.push_back(iter);
          } else if (n_split == 2) {
            space_2.push_back(iter);
          } else {
            ICHECK_EQ(n_split, 3);
            space_3.push_back(iter);
          }
        }
      }
    } else {
      LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
    }
  }

  if (n_split == 3) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2, &space_3);
  } else if (n_split == 2) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2);
  } else {
    ConcatenateMove(&tmp_order, &space_0, &space_1);
  }
  tmp_s.reorder(stage_id, tmp_order);
  return tmp_s;
}

// Return whether a state has nested parallel, which is invalid on CPUs
bool HasNestedParallel(const State& state) {
  std::function<void(int stage_id, size_t*)> count_parallel_ct;

  count_parallel_ct = [&state, &count_parallel_ct](int stage_id, size_t* parallel_ct) {
    const Stage& stage = state->stages[stage_id];

    if (stage->compute_at == ComputeAtKind::kInlined) {
      return;
    }

    for (size_t i = 0; i < stage->iters.size(); ++i) {
      if (stage->iters[i]->annotation == IteratorAnnotation::kParallel) {
        (*parallel_ct)++;
      }

      IterKey iter_key(stage_id, i);
      auto pair = state->attach_map->iter_to_attached_stages.find(iter_key);
      if (pair != state->attach_map->iter_to_attached_stages.end()) {
        for (const auto& attach_stage_id : pair->second) {
          count_parallel_ct(attach_stage_id, parallel_ct);
        }
      }
    }
  };

  for (size_t stage_id = 0; stage_id < state->stages.size(); ++stage_id) {
    size_t parallel_ct = 0;

    if (state->stages[stage_id]->compute_at == ComputeAtKind::kRoot) {
      count_parallel_ct(stage_id, &parallel_ct);
      if (parallel_ct >= 2) {
        return true;
      }
    }
  }

  return false;
}

bool CheckAndVerify(const SearchTask& task, const State& state) {
  te::Schedule sch;
  Array<te::Tensor> tensors;

  // NOTE: Currently, feature extraction with and without layout rewrite
  // returns the same feature vector, so we do not turn on layout rewrite here.
  // In the future, we can improve the feature extraction to reflect this difference.
  std::tie(sch, tensors) = task->compute_dag.ApplySteps(state->transform_steps);
  sch = sch.normalize_for_feature_extraction();
  auto bounds = te::InferBound(sch);

  try {
    auto stmt = te::ScheduleOps(sch, bounds, false);
    Map<te::Tensor, te::Buffer> out_binds;
    Array<ObjectRef> out_arg_list;
    bool compact = te::VerifyCompactBuffer(stmt);
    const std::string& name = "main";
    GlobalVar global_var(name);

    // Copied from driver_api.cc::lower
    auto pass_ctx = tvm::transform::PassContext::Current();
    GetBinds(tensors, compact, std::unordered_map<te::Tensor, te::Buffer>(), &out_binds,
             &out_arg_list);
    tir::PrimFunc f = te::SchedulePostProcToPrimFunc(out_arg_list, std::move(stmt), out_binds);
    f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

    bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();
    bool disable_vectorize =
        pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
    bool instrument_bound_checkers =
        pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value();

    if (noalias) {
      f = WithAttr(std::move(f), "tir.noalias", Bool(true));
    }
    auto mod = IRModule(Map<GlobalVar, BaseFunc>({{global_var, f}}));

    if (IsGPUTask(task)) {
      auto pass_list = Array<tvm::transform::Pass>();
      // Phase 0
      pass_list.push_back(tir::transform::InjectPrefetch());
      pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
      // Phase 1
      pass_list.push_back(tir::transform::NarrowDataType(32));
      pass_list.push_back(tir::transform::Simplify());
      pass_list.push_back(tir::transform::VectorizeLoop(!disable_vectorize));
      pass_list.push_back(tir::transform::InjectVirtualThread());
      pass_list.push_back(tir::transform::StorageRewrite());
      pass_list.push_back(tir::transform::Simplify());
      tvm::Map<String, tvm::PrimExpr> gpu_params{
          {"max_shared_memory_per_block", task->hardware_params->max_shared_memory_per_block},
          {"max_local_memory_per_block", task->hardware_params->max_local_memory_per_block},
          {"max_threads_per_block", task->hardware_params->max_threads_per_block},
          {"max_vector_bytes", task->hardware_params->vector_unit_bytes},
          {"max_vthread", task->hardware_params->max_vthread_extent},
      };
      pass_list.push_back(tir::transform::VerifyGPUCode(gpu_params));
      const auto& optimize = tir::transform::Sequential(pass_list);
      optimize(mod);
    }
    const auto& optimize =
        tir::transform::Sequential(Array<tvm::transform::Pass>{tir::transform::Simplify()});
    mod = optimize(std::move(mod));
  } catch (Error& e) {
    return false;
  }
  return true;
}


void PruneInvalidState(const SearchTask& task, Array<State>* states, bool verify) {
  bool is_gpu = IsGPUTask(task);
  std::vector<int> valid(states->size(), 0);
  if (is_gpu && verify) {
    support::parallel_for(0, states->size(),
                        [&task, &states, &valid](int i) {
                          valid[i] = CheckAndVerify(task, (*states)[i]);
                        });
  }
  size_t pt = 0;
  for (size_t i = 0; i < states->size(); ++i) {
    if (!(*states)[i].defined()) {
      continue;
    }
    if (!is_gpu && HasNestedParallel((*states)[i])) {
      continue;
    }
    if (is_gpu && verify && !valid[i]) {
      continue;
    }
    if (i != pt) {
      states->Set(pt, (*states)[i]);
    }
    pt++;
  }

  if (pt == 0) {
    LOG(FATAL) << "Internal error: All states are invalid.";
  } else {
    // LOG(INFO) << "Valid ratio: " << pt << "/" << states->size();
    states->resize(pt);
  }
}

/********** SplitFactorizationMemo **********/
const Array<Array<Integer>>& SplitFactorizationMemo::GetFactorizationSchemes(
    int extent, int n_lengths, int max_innermost_factor) {
  QueryKey key = std::make_tuple(extent, n_lengths, max_innermost_factor);
  const auto& it = memory_.find(key);
  if (it != memory_.end()) {
    return it->second;
  }

  tmp_stack_ = Array<Integer>(n_lengths, Integer());
  results_ = &memory_[key];
  n_lengths_ = n_lengths;

  DfsEnumerate(0, extent, max_innermost_factor);

  return *results_;
}

void SplitFactorizationMemo::DfsEnumerate(int now, int remaining_length, int max_innermost_factor) {
  if (now == n_lengths_) {
    if (tmp_stack_.back().as<IntImmNode>()->value <= max_innermost_factor) {
      results_->push_back(tmp_stack_);
    }
  } else {
    for (const auto& f : GetFactors(remaining_length)) {
      tmp_stack_.Set(now, Integer(f));
      DfsEnumerate(now + 1, remaining_length / f, max_innermost_factor);
    }
  }
}

const std::vector<int>& SplitFactorizationMemo::GetFactors(int n) {
  auto it = factor_memory_.find(n);
  if (it != factor_memory_.end()) {
    return it->second;
  }

  std::vector<int>& res = factor_memory_[n];
  int step = n % 2 == 0 ? 1 : 2;
  for (size_t i = 1; i < static_cast<size_t>(std::sqrt(n)) + 1; i += step) {
    if (n % i == 0) {
      res.push_back(i);
      if (n / i != i) {
        res.push_back(n / i);
      }
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}

/********** Utils interface API for ffi **********/

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsGetConsumers")
    .set_body_typed([](const SearchTask& task, const State& state, int stage_id) {
      const std::set<int>& consumers = GetConsumers(task, state, stage_id);
      tvm::Map<IntImm, IntImm> ret;
      for (const auto& i : consumers) {
        ret.Set(Integer(i), Integer(i));
      }
      return ret;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsIsElementwiseMatch")
    .set_body_typed([](const SearchTask& task, const State& state, int stage_id,
                       int target_stage_id) {
      return ElementwiseMatch(task, state, stage_id, target_stage_id);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsIsTiled")
    .set_body_typed([](const Stage& stage) { return IsTiled(stage); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCacheReadStage")
    .set_body_typed([](const State& s, int stage_id) { return HasCacheReadStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCacheWriteStage")
    .set_body_typed([](const State& s, int stage_id) { return HasCacheWriteStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasRfactorStage")
    .set_body_typed([](const State& s, int stage_id) { return HasRfactorStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCrossThreadReduction")
    .set_body_typed([](const State& s, int stage_id) {
      return HasCrossThreadReduction(s, stage_id);
    });

}  // namespace auto_scheduler
}  // namespace tvm
