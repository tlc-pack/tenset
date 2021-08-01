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
 * \file ansor/feature.cc
 * \brief Feature extraction for the cost model
 */

#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/runtime/registry.h>
#include <tvm/arith/analyzer.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>
#include "search_policy/utils.h"
#include "utils.h"
#include <chrono>
#include <tvm/auto_scheduler/feature.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/measure_record.h>


namespace tvm {
/* Import the function from driver_api.cc */
extern void GetBinds(const Array<te::Tensor>& args, bool compact,
                     const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                     Map<te::Tensor, tir::Buffer>* out_binds, Array<ObjectRef>* out_arg_list);
}  // namespace tvm


namespace tvm {
namespace auto_scheduler {

using namespace tvm::tir;
using arith::ConstIntBound;
using arith::Analyzer;


template <class T>
using BufferMap = std::unordered_map<Buffer, T, ObjectHash, ObjectEqual>;

// The number of samples to extract for arithmetic intensity curves
static const int ARITH_INTENSITY_CURVE_SAMPLE_N = 10;

// Annotation position encoding
enum class AnnotationPosType : int {
  kPosNone = 0,           // Does not have this kind of annotation
  kPosInnerSpatial = 1,   // The annotated iterator is the innermost spatial iterator
  kPosMiddleSpatial = 2,  // The annotated iterator is a middle spatial iterator
  kPosOuterSpatial = 3,   // The annotated iterator is the outermost spatial iterator
  kPosInnerReduce = 4,    // The annotated iterator is the innermost reduce iterator
  kPosMiddleReduce = 5,   // The annotated iterator is a middle reduce iterator
  kPosOuterReduce = 6,    // The annotated iterator is the outermost reduce iterator
  kPosMixed = 7           // The annotated iterator is a mixed space and reduce iterator
};

// Buffer access type
enum class BufferAccessType : int { kRead = 0, kWrite = 1, kReadWrite = 2, kUnknownRW = 3 };

// Accesses to a buffer
struct BufferAccess {
  // data reuse type
  BufferAccessType acc_type{BufferAccessType::kUnknownRW};
  // Use a two-dimensional array to store multiple multi-dimensional accesses.
  // The innermost vector stores the multi-dimensional indices of one access.
  std::vector<std::vector<PrimExpr>> indices;
};

// Data reuse type
enum class ReuseType : int { kLoopMultipleRead = 0, kSerialMultipleReadWrite = 1, kNoReuse = 2 };

// Feature for an access of a buffer
struct BufferAccessFeature {
  std::string buffer_name;
  BufferAccessType acc_type;
  float bytes;
  float unique_bytes;
  float lines;
  float unique_lines;
  ReuseType reuse_type;
  float reuse_dis_iter;    // reuse distance in iterator number
  float reuse_dis_bytes;   // reuse distance in total touched bytes
  float reuse_ct;          // reuse times
  float bytes_d_reuse_ct;
  float unique_bytes_d_reuse_ct;
  float lines_d_reuse_ct;
  float unique_lines_d_reuse_ct;
  float stride;
};

// Accesses to a buffer
struct BufferAccess {
  BufferAccessType acc_type{kUnknownRW};
  std::vector<std::vector<PrimExpr> > indices;
};

static const int NODE_FEATURE_LENGTH = 84;
static const int EDGE_FEATURE_LENGTH = 4;



struct Edge {
  int src;
  int dst;
  float feature[EDGE_FEATURE_LENGTH];
};

// {"stmtnode": 0, ""}
struct Node {
  int node_type;
  int id;
  float feature[NODE_FEATURE_LENGTH];
};


// Return the min of a for loop
int64_t GetLoopMin(const ForNode* node) {
  auto pint = node->min.as<IntImmNode>();
  if (pint != nullptr) {
    return pint->value;
  } else {
    return 0;
  }
}


// Count math ops in an expr
class MathOpCounter : public StmtExprVisitor {
 public:
#define VisitBinary(Type, float_ct, int_ct) \
  void VisitExpr_(const Type* op) final {   \
    if (op->a.dtype().is_float()) {          \
      float_ct++;                           \
    } else {                                \
      int_ct++;                             \
    }                                       \
    StmtExprVisitor::VisitExpr_(op);        \
  }                                         \

  VisitBinary(AddNode, float_addsub, int_addsub);
  VisitBinary(SubNode, float_addsub, int_addsub);
  VisitBinary(MulNode, float_mul, int_mul);
  VisitBinary(DivNode, float_divmod, int_divmod);
  VisitBinary(ModNode, float_divmod, int_divmod);
  VisitBinary(FloorDivNode, float_divmod, int_divmod);
  VisitBinary(FloorModNode, float_divmod, int_divmod);
  VisitBinary(MaxNode, float_cmp, int_cmp);
  VisitBinary(MinNode, float_cmp, int_cmp);
  VisitBinary(EQNode, float_cmp, int_cmp);
  VisitBinary(NENode, float_cmp, int_cmp);
  VisitBinary(LTNode, float_cmp, int_cmp);
  VisitBinary(LENode, float_cmp, int_cmp);
  VisitBinary(GTNode, float_cmp, int_cmp);
  VisitBinary(GENode, float_cmp, int_cmp);

  void VisitExpr_(const AndNode* op) final { bool_op++; StmtExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const OrNode* op)  final { bool_op++; StmtExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const NotNode* op) final { bool_op++; StmtExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const SelectNode* op) final { select_op++; StmtExprVisitor::VisitExpr_(op); }

  void VisitExpr_(const CallNode* op) final {
    auto* pop = op->op.as<OpNode>();
    CHECK(pop != nullptr);
    auto effect_kind = op_call_effect_[GetRef<Op>(pop)];
    bool is_pure = effect_kind == CallEffectKind::kPure || effect_kind == CallEffectKind::kExprAnnotation;

    if (is_pure) {
      if (op->dtype.is_float()) {
        float_math_func++;
      } else {
        int_math_func++;
      }
    } else {
      if (op->dtype.is_float()) {
        float_other_func++;
      } else {
        int_other_func++;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  // todo(lmzheng): detect mad
  size_t float_mad{0}, float_addsub{0}, float_mul{0}, float_divmod{0},
         float_cmp{0}, float_math_func{0}, float_other_func{0};
  size_t int_mad{0}, int_addsub{0}, int_mul{0}, int_divmod{0},
         int_cmp{0}, int_math_func{0}, int_other_func{0};
  size_t bool_op{0}, select_op{0};

  OpAttrMap<TCallEffectKind> op_call_effect_ = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
};


// Extract all buffer accesses in an expr
class BufferAccessExtractor : public StmtExprVisitor {
 public:
  void ExtractReads(const PrimExpr& expr) {
    this->VisitExpr(expr);
  }

  void InsertAccess(const Buffer& buf, BufferAccessType acc_type,
      const Array<PrimExpr>& indices) {
    BufferAccess& acc = buf_accesses[buf];
    acc.acc_type = acc_type;
    acc.indices.push_back(std::vector<PrimExpr>(indices.begin(), indices.end()));
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    BufferAccess& acc = buf_accesses[op->buffer];
    switch (acc.acc_type) {
      case kRead:
        break;
      case kWrite:
        acc.acc_type = kReadWrite; break;
      case kReadWrite:
        break;
      case kUnknownRW:
      default:
        acc.acc_type = kRead; break;
    }

    if (acc.acc_type != kReadWrite) {
      // If a buffer is both read and written, in the tvm DSL, it must be a update,
      // so the indices should be the same. Then we can skip appending indices for it.
      // Otherwise we do the following.
      buf_accesses[op->buffer].indices.push_back(
          std::vector<PrimExpr>(op->indices.begin(), op->indices.end()));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  BufferMap<BufferAccess> buf_accesses;
};

// Compute coefficient for an loop iterator in an expression
// Note: we use a approximation strategy to find coefficient.
// Hopefully, it is faster than DetectLinearEquation and can handle more cases (non-linear)
class CoefficientExtractor : public StmtExprVisitor {
 public:
  void VisitExpr_(const MulNode *node) final {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {

      if (!visited_add) {
//        std::cout << "mul" << std::endl;
        if (auto a = node->a.as<IntImmNode>()) {
          visited_mul = true;
          stride = a->value;
        } else if (auto b = node->b.as<IntImmNode>()) {
          visited_mul = true;
          stride = b->value;
        }
      }
    }
  }

  void VisitExpr_(const AddNode *node) final {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_mul) {
//        std::cout << "add" << std::endl;
        visited_add = true;
        stride = 1;
      }
    }
  }

  void VisitExpr_(const VarNode *node) final {
    if (node == var_) {
      visited_var = true;
//      std::cout << "var" << std::endl;

      // This is a magic default stride in case our approximation strategy fails
      stride = 2;
    }
  }

  int ExtractCoefficient(const PrimExpr& expr, const VarNode* var) {
    //chloe
    stride = 0;
    visited_var = visited_mul = visited_add = false;
    var_ = var;
    this->VisitExpr(expr);

    if (visited_var && !visited_mul && !visited_add) {
      return 1;
    } else {
      return stride;
    }
  }

  bool visited_var{false};
  bool visited_mul{false};
  bool visited_add{false};
  int stride{0};

 private:
  const VarNode* var_{nullptr};
};


// shifted log to incorporate the property that slog(0) = 0
inline float slog(float x) {
  return x < 0 ? -std::log2(-x+1) : std::log2(x+1);
}

inline float slog2(float x) {
  return slog(slog(x));
}


class NodeGather : public StmtExprVisitor {
 public:
  void VisitStmt(const Stmt& n) {
//    int count = static_cast<int>(node_to_index.size());
    int primExprType = 0;
    int64_t forExtent = 0;
    int64_t forMin = 0;
    int forType = 0;
    if (n->IsInstance<LetStmtNode>()) {
      primExprType = 0;
    } else if (n->IsInstance<AttrStmtNode>()) {
      primExprType = 1;
    } else if (n->IsInstance<IfThenElseNode>()) {
      primExprType = 2;
    } else if (n->IsInstance<ForNode>()) {
      primExprType = 3;
      forExtent = GetLoopExtent(n.as<ForNode>());
      forMin = GetLoopMin(n.as<ForNode>());
      forType = (int)n.as<ForNode>()->for_type;
    } else if (n->IsInstance<AllocateNode>()) {
      primExprType = 4;
    } else if (n->IsInstance<StoreNode>()) {
      primExprType = 5;
    } else if (n->IsInstance<BufferStoreNode>()) {
      primExprType = 6;
    } else if (n->IsInstance<BufferRealizeNode>()) {
      primExprType = 7;
    } else if (n->IsInstance<AssertStmtNode>()) {
      primExprType = 8;
    } else if (n->IsInstance<ProducerStoreNode>()) {
      primExprType = 9;
    } else if (n->IsInstance<ProducerRealizeNode>()) {
      primExprType = 10;
    } else if (n->IsInstance<PrefetchNode>()) {
      primExprType = 11;
    } else if (n->IsInstance<SeqStmtNode>()) {
      primExprType = 12;
    } else if (n->IsInstance<EvaluateNode>()) {
      primExprType = 13;
    }
    if (node_to_index.find(n.get()) == node_to_index.end()) {
      Node newNode = {
          0,
          count,
          {0}
      };
      newNode.feature[0] = 0;
      newNode.feature[primExprType+1] = 1;
      if (primExprType == 3) {
        newNode.feature[35] = slog2(forExtent);
        newNode.feature[36] = slog2(forMin);
        newNode.feature[37+forType] = 1;

        const ForNode* node = n.as<ForNode>();

        int64_t loop_extent = GetLoopExtent(node);
        if (node->for_type == ForType::Vectorized) {
          vec_for_stack.push_back(node);
        } else if (node->for_type == ForType::Unrolled) {
          unroll_for_stack.push_back(node);
        } else if (node->for_type == ForType::Parallel) {
          parallel_for_stack.push_back(node);
        }

        outer_loop_prod *= loop_extent;
        for_loop_stack.push_back(node);

        node_list.push_back(newNode);
        node_to_index[n.get()] = count;
        count++;

        StmtExprVisitor::VisitStmt_(node);
        for_loop_stack.pop_back();
        outer_loop_prod /= loop_extent;

        if (node->for_type == ForType::Vectorized) {
          vec_for_stack.pop_back();
        } else if (node->for_type == ForType::Unrolled) {
          unroll_for_stack.pop_back();
        } else if (node->for_type == ForType::Parallel) {
          parallel_for_stack.pop_back();
        }

      }

      if (primExprType == 6) {

         const BufferStoreNode* node = n.as<BufferStoreNode>();
         MathOpCounter mathops;
         mathops(node->value);

         std::vector<BufferAccessFeature> acc_feas;
         BufferAccessExtractor buf_extractor;
         buf_extractor.InsertAccess(node->buffer, kWrite, node->indices);
         buf_extractor.ExtractReads(node->value);

         Analyzer ana;
        for (auto x : for_loop_stack) {
          ana.Bind(x->loop_var, Range::make_by_min_extent(x->min, 1), true);
        }

        std::vector<int> tmp_region;
        for (int i = static_cast<int>(for_loop_stack.size()) - 1; i >= 0; i--) {
          const ForNode* p_for = for_loop_stack[i];

          ana.Bind(p_for->loop_var,
                   Range::make_by_min_extent(for_loop_stack[i]->min, for_loop_stack[i]->extent), true);

          BufferMap<std::vector<std::tuple<BufferAccessType, int64_t, int> > >&
              buffer_regions_map = for_touch_regions[p_for];

          for (const auto &x : buf_extractor.buf_accesses) {
            const Buffer& t = x.first;
            const BufferAccess& acc = x.second;

            ComputeRegion(acc.indices, &ana, &tmp_region);
            int64_t touched_size = ElementProduct(tmp_region);
            buffer_regions_map[t].push_back(std::make_tuple(acc.acc_type,
                        touched_size, t->dtype.bytes()));
          }
        }

         for (const auto &x : buf_extractor.buf_accesses) {
          const Buffer& t = x.first;
          const BufferAccess& acc = x.second;


          std::vector<int> int_shape;
          for (const auto& dim : t->shape) {
            int_shape.push_back(GetIntImm(dim));
          }

          size_t ele_bytes = t->dtype.bytes();

          // calculate bytes
          float bytes = outer_loop_prod * ele_bytes;
          float unique_bytes;

          // calculate cache lines
          int64_t stride;
          float lines;
          float unique_lines;

          if (for_loop_stack.empty()) {
            unique_bytes = ele_bytes;
            stride = 0;
            lines = 1.0f;
            unique_lines = 1.0f;
          } else {
            unique_bytes = std::get<1>(for_touch_regions[for_loop_stack.front()][t].front())
                * ele_bytes;

            stride = 0;
            int64_t reduce_ratio = 1;

            int i;
            for (i = static_cast<int>(for_loop_stack.size()) - 1; i >= 0; i--) {
              stride = ComputeStride(acc.indices, int_shape, for_loop_stack[i]->loop_var.get());
              if (stride != 0) {
                break;
              }

              reduce_ratio *= GetLoopExtent(for_loop_stack.back());
            }

            lines = outer_loop_prod / reduce_ratio *
                std::min(1.0f, 1.0f * stride * ele_bytes / cache_line_size_);
            lines = std::max(lines, 1.0f);

    //        // convert `stride` back to the stride of the innermost iterator
            stride = (i == static_cast<int>(for_loop_stack.size()) - 1 ? stride : 0);

            float n_continuous = ele_bytes;
            for (int i = std::min(static_cast<int>(tmp_region.size()) - 1,
                                  static_cast<int>(int_shape.size()) - 1); i >= 0; i--)  {
              if (tmp_region[i] == int_shape[i]) {
                n_continuous *= tmp_region[i];
                break;
              }
            }
            unique_lines = unique_bytes / std::min(n_continuous,
                                                   static_cast<float>(cache_line_size_));
            unique_lines = std::max(unique_lines, 1.0f);
          }


          ReuseType reuse_type;
          float reuse_dis_iter, reuse_dis_bytes, reuse_ct;
          std::tie(reuse_type, reuse_dis_iter, reuse_dis_bytes, reuse_ct) =
              ComputeReuse(t, acc.indices, for_loop_stack, for_touch_regions);

          acc_feas.emplace_back();
          BufferAccessFeature& acc_fea = acc_feas.back();

          acc_fea.acc_type = acc.acc_type;
          acc_fea.stride = stride;
          acc_fea.bytes = bytes;
          acc_fea.unique_bytes = unique_bytes;
          acc_fea.lines = lines;
          acc_fea.unique_lines = unique_lines;
          acc_fea.reuse_type = reuse_type;
          acc_fea.reuse_dis_iter = reuse_dis_iter;
          acc_fea.reuse_dis_bytes = reuse_dis_bytes;
          acc_fea.reuse_ct = reuse_ct;
        }


        std::vector<std::pair<float, float> > buf_order_key;
        for (const auto& acc_fea : acc_feas) {
          buf_order_key.emplace_back(acc_fea.lines, acc_fea.bytes);
        }
        std::vector<int> buf_order(buf_order_key.size());
        std::iota(buf_order.begin(), buf_order.end(), 0);

        auto cmp = [&buf_order_key](int l, int r) {
          return buf_order_key[l].first > buf_order_key[r].first
              || (buf_order_key[l].first == buf_order_key[r].first
                  && buf_order_key[l].second > buf_order_key[r].second);
        };
        std::sort(buf_order.begin(), buf_order.end(), cmp);
        int n_bufs = std::min(max_n_bufs, static_cast<int>(buf_order.size()));
        buf_order.resize(n_bufs);


        int i = 42;
        for (int idx : buf_order) {
          const auto& acc_fea = acc_feas[idx];
          for (int j = 0; j <= kReadWrite; ++j) {
            newNode.feature[i++] = (j == acc_fea.acc_type);
          }
          newNode.feature[i++] = slog2(acc_fea.bytes);
          newNode.feature[i++] = slog2(acc_fea.unique_bytes);
          newNode.feature[i++] = slog2(acc_fea.lines);
          newNode.feature[i++] = slog2(acc_fea.unique_lines);
          for (int j = 0; j <= kNoReuse; ++j) {
            newNode.feature[i++] = (acc_fea.reuse_type == j);
          }

          newNode.feature[i++] = slog2(acc_fea.reuse_dis_iter);
          newNode.feature[i++] = slog2(acc_fea.reuse_dis_bytes);
          newNode.feature[i++] = slog2(acc_fea.reuse_ct);
          newNode.feature[i++] = slog2(acc_fea.stride);

        }
        // - fill padding
        for (int k = 0; k < max_n_bufs - n_bufs; ++k) {
          for (int j = 0; j <= kReadWrite; ++j) {  // 3
            newNode.feature[i++] = 0.0f;
          }
          newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f;
          for (int j = 0; j <= kNoReuse; ++j) {   // 3
            newNode.feature[i++] = 0.0f;
          }
          newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f;
        }

        }
          if (primExprType != 3) {
              node_list.push_back(newNode);
              node_to_index[n.get()] = count;
              count++;
          }
      }
    if (primExprType != 3) {
        StmtExprVisitor::VisitStmt(n);
    }
  }

  void VisitExpr(const PrimExpr& n) {
    //    int count = static_cast<int>(node_to_index.size());
    int primExprType = 0;

    if (n->IsInstance<CallNode>()) {
      primExprType = 0;
    } else if (n->IsInstance<AddNode>()) {
      primExprType = 1;
    } else if (n->IsInstance<SubNode>()) {
      primExprType = 2;
    } else if (n->IsInstance<MulNode>()) {
      primExprType = 3;
    } else if (n->IsInstance<DivNode>()) {
      primExprType = 4;
    } else if (n->IsInstance<ModNode>()) {
      primExprType = 5;
    } else if (n->IsInstance<FloorDivNode>()) {
      primExprType = 6;
    } else if (n->IsInstance<FloorModNode>()) {
      primExprType = 7;
    } else if (n->IsInstance<MinNode>()) {
      primExprType = 8;
    } else if (n->IsInstance<MaxNode>()) {
      primExprType = 9;
    } else if (n->IsInstance<EQNode>()) {
      primExprType = 10;
    } else if (n->IsInstance<NENode>()) {
      primExprType = 11;
    } else if (n->IsInstance<LTNode>()) {
      primExprType = 12;
    } else if (n->IsInstance<LENode>()) {
      primExprType = 13;
    } else if (n->IsInstance<GTNode>()) {
      primExprType = 14;
    } else if (n->IsInstance<GENode>()) {
      primExprType = 15;
    } else if (n->IsInstance<AndNode>()) {
      primExprType = 16;
    } else if (n->IsInstance<OrNode>()) {
      primExprType = 17;
    } else if (n->IsInstance<ReduceNode>()) {
      primExprType = 18;
    } else if (n->IsInstance<CastNode>()) {
      primExprType = 19;
    } else if (n->IsInstance<NotNode>()) {
      primExprType = 20;
    } else if (n->IsInstance<SelectNode>()) {
      primExprType = 21;
    } else if (n->IsInstance<RampNode>()) {
      primExprType = 22;
    } else if (n->IsInstance<BroadcastNode>()) {
      primExprType = 23;
    } else if (n->IsInstance<ShuffleNode>()) {
      primExprType = 24;
    } else if (n->IsInstance<IntImmNode>()) {
      primExprType = 25;
    } else if (n->IsInstance<FloatImmNode>()) {
      primExprType = 26;
    } else if (n->IsInstance<StringImmNode>()) {
      primExprType = 27;
    } else if (n->IsInstance<VarNode>()) {
      primExprType = 28;
    } else if (n->IsInstance<SizeVarNode>()) {
      primExprType = 29;
    } else if (n->IsInstance<BufferLoadNode>()) {
      primExprType = 30;
    } else if (n->IsInstance<ProducerLoadNode>()) {
      primExprType = 31;
    } else if (n->IsInstance<LoadNode>()) {
      primExprType = 32;
    } else if (n->IsInstance<LetNode>()) {
      primExprType = 33;
    }

    if (node_to_index.find(n.get()) == node_to_index.end()) {
      Node newNode = {
          1,
          count,
          {0}
      };
      newNode.feature[0] = 1;
      newNode.feature[primExprType+1] = 1;

      if (primExprType == 25) {
        newNode.feature[41] = slog2(n.as<IntImmNode>()->value);
      }
      if (primExprType == 26) {
        newNode.feature[41] = slog2(n.as<FloatImmNode>()->value);
      }
      node_list.push_back(newNode);
      node_to_index[n.get()] = count;
      count++;
    }

    StmtExprVisitor::VisitExpr(n);
  }

  std::unordered_map<const Object*, int> node_to_index;
  std::vector<Node> node_list;
  int count = 0;
  int max_n_bufs = 3;

  std::unordered_map<std::string, int> buffer_lookup;
  int buffer_count = 0;

  float outer_loop_prod = 1.0f;

  std::vector<const ForNode*> for_loop_stack;
  std::vector<const ForNode*> parallel_for_stack;
  std::vector<const ForNode*> vec_for_stack;
  std::vector<const ForNode*> unroll_for_stack;

  std::unordered_map<const ForNode*, BufferMap<std::vector<
  std::tuple<BufferAccessType, int64_t, int> > > > for_touch_regions;
  private:
    const int cache_line_size_ = 64;

};


class EdgeGather: public StmtExprVisitor {
 public:
  void VisitStmt_(const AttrStmtNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = node_to_index[node->value.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);

    dst_idx = node_to_index[node->body.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitStmt_(const ForNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = node_to_index[node->min.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[2] = 1;
    edge_list.push_back(newEdge);

    dst_idx = node_to_index[node->loop_var.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);

    dst_idx = node_to_index[node->body.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);

    dst_idx = node_to_index[node->extent.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[2] = 1;
    edge_list.push_back(newEdge);


    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitStmt_(const BufferStoreNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = node_to_index[node->value.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    for (std::size_t i = 0; i < node->indices.size(); i++) {
      dst_idx = node_to_index[node->indices[i].get()];
      newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }

    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitStmt_(const BufferRealizeNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = node_to_index[node->condition.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[node->body.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    for (const Range &range : node->bounds) {
      dst_idx = node_to_index[range->min.get()];
      newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
      dst_idx = node_to_index[range->extent.get()];
      newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }

    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitStmt_(const SeqStmtNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = 0;
    for (const Stmt &s : node->seq) {
      dst_idx = node_to_index[s.get()];
      Edge newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }
    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode *op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = 0;
    for (const PrimExpr &primExpr : op->indices) {
      dst_idx = node_to_index[primExpr.get()];
      Edge newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const AddNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MulNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const DivNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ModNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const FloorDivNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const FloorModNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MinNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MaxNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const EQNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const NENode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const LTNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const LENode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const GTNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const GENode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const AndNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const OrNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ReduceNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx;
    for (int i = 0; op->source.size(); i++) {
      dst_idx = node_to_index[op->source[i].get()];
      Edge newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }
    for (int i = 0; op->axis.size(); i++) {
      dst_idx = node_to_index[op->axis[i].get()];
      Edge newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }
    dst_idx = node_to_index[op->condition.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  std::vector<Edge> edge_list;
  std::unordered_map<const Object*, int> node_to_index;

};


TVMByteArray SerializeGraph(std::vector<std::vector<Edge> >& edge_list,
                            std::vector<std::vector<Node> >& node_list,
                            std::vector<float>&& normalized_throughputs,
                            std::vector<int>&& task_ids,
                            std::vector<char>* out_data) {
  /* Serialization format
   * {
   *   size_vector;
   *   Edge edges[n_edges];
   *   Node nodes[n_nodes];
   *   normalized_throughputs;
   *   task_ids;
   * }
   */

  std::vector<int> size_vector;
  size_t total_bytes = 0;
  int n = edge_list.size();

  // serialize sizes
  size_t size_vector_size = 1 + n + n + 2;
  total_bytes += size_vector_size * sizeof(int);

  size_vector.reserve(size_vector_size);
  size_vector.push_back(edge_list.size());
  for (const auto& x : edge_list) {
    size_vector.push_back(static_cast<int>(x.size())*2);
    total_bytes += sizeof(Edge) * x.size() * 2;
  }
  for (const auto& x : node_list) {
    size_vector.push_back(static_cast<int>(x.size()));
    total_bytes += sizeof(Node) * x.size();
  }

  size_vector.push_back(static_cast<int>(normalized_throughputs.size()));
  total_bytes += sizeof(float) * normalized_throughputs.size();
  size_vector.push_back(static_cast<int>(task_ids.size()));
  total_bytes += sizeof(int) * task_ids.size();

  CHECK_EQ(size_vector.size(), size_vector_size);

  out_data->reserve(total_bytes);
  char* ptr = out_data->data();

  // serialize size_vector
  memmove(ptr, reinterpret_cast<char*>(size_vector.data()), size_vector.size() * sizeof(int));
  ptr += size_vector.size() * sizeof(int);

  // serialize edge list
  for (auto& x : edge_list) {
    for (Edge& edge : x) {
      memcpy(ptr, &edge.src, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, &edge.dst, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, edge.feature, sizeof(float) * EDGE_FEATURE_LENGTH);
      ptr += sizeof(float) * EDGE_FEATURE_LENGTH;
    }
//    x.clear();
    //adding backward edges
    for (Edge& edge : x) {
      memcpy(ptr, &edge.dst, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, &edge.src, sizeof(int));
      ptr += sizeof(int);
      edge.feature[0] = 0;
      edge.feature[1] = 1;
      memcpy(ptr, edge.feature, sizeof(float) * EDGE_FEATURE_LENGTH);
      ptr += sizeof(float) * EDGE_FEATURE_LENGTH;
    }
    x.clear();
  }

  // serialize node list
  for (auto& x : node_list) {
    for (Node& node : x) {
      memcpy(ptr, &node.node_type, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, &node.id, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, node.feature, sizeof(float) * NODE_FEATURE_LENGTH);
      ptr += sizeof(float) * NODE_FEATURE_LENGTH;
    }
    x.clear();
  }

  // serialize normalized_throughputs
  memmove(ptr, reinterpret_cast<char*>(normalized_throughputs.data()),
          normalized_throughputs.size() * sizeof(float));
  ptr += normalized_throughputs.size() * sizeof(float);

  // serialize task_ids
  memmove(ptr, reinterpret_cast<char*>(task_ids.data()), task_ids.size() * sizeof(int));
  ptr += task_ids.size() * sizeof(int);

  CHECK_EQ(ptr - out_data->data(), total_bytes);

  return TVMByteArray{out_data->data(), total_bytes};
}

//int i = 0;
void GetGraph(const State& state,
                     const SearchTask& task,
                     int max_n_bufs,
                     std::vector<Node>* node_list,
                     std::vector<Edge>* edge_list) {
  te::Schedule sch;
  Array<te::Tensor> tensors;

  std::tie(sch, tensors) = task->compute_dag.ApplySteps(state->transform_steps);
  sch = sch.normalize(true);
  auto bounds = te::InferBound(sch);

  auto stmt = te::ScheduleOps(sch, bounds, false);
  Map<te::Tensor, te::Buffer> out_binds; Array<ObjectRef> out_arg_list;
  bool compact = te::VerifyCompactBuffer(stmt);
  const std::string& name = "main";
  GlobalVar global_var(name);

  // Copied from driver_api.cc::lower
  auto pass_ctx = tvm::transform::PassContext::Current();
  GetBinds(tensors, compact, std::unordered_map<te::Tensor, te::Buffer>(),
           &out_binds, &out_arg_list);
  tir::PrimFunc f = te::SchedulePostProcToPrimFunc(out_arg_list, std::move(stmt), out_binds);
  f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

  auto mod = IRModule(Map<GlobalVar, BaseFunc>({{global_var, f}}));

  const auto& optimize = tir::transform::Sequential(
      Array<tvm::transform::Pass>{tir::transform::Simplify()});
  mod = optimize(std::move(mod));
  const auto& it = mod->functions.find(global_var);
  CHECK(it != mod->functions.end());
  const auto& prim_func = (*it).second.as<PrimFuncNode>();

//  if (i == 0) {
//    std::cout << prim_func->body << std::endl;
//    i++;
//  }

  NodeGather nodeGather;
  nodeGather(prim_func->body);

  for (auto node : nodeGather.node_list) {
    node_list->push_back(node);
  }

  EdgeGather edgeGather;
  edgeGather.node_to_index = nodeGather.node_to_index;
  edgeGather(prim_func->body);
  for (auto edge : edgeGather.edge_list) {
    edge_list->push_back(edge);
  }

}

void GetGraphFromStates(const Array<State>& states,
                               const std::vector<SearchTask>& tasks,
                               int max_n_bufs,
                               std::vector<std::vector<Node> >* node_list,
                               std::vector<std::vector<Edge> >* edge_list) {
  // extract features
  node_list->assign(states.size(), std::vector<Node>());
  edge_list->assign(states.size(), std::vector<Edge>());

  std::atomic<int> error_ct(0);

  ThreadPool& pool = ThreadPool::Global();
  pool.BeginBatch(static_cast<int>(states.size()) - 0);
  for (size_t i = 0; i < states.size(); ++i) {
    pool.Enqueue(GetGraph, states[i], tasks[i],
                 max_n_bufs, &(*node_list)[i], &(*edge_list)[i]);
  }
  pool.WaitBatch();

  if (error_ct > 0) {
    std::cerr << "Encountered " << error_ct
              << " errors during feature extraction. which are safely ignored." << std::endl;
  }
}

void GetGraphFromStates(const Array<State>& states,
                               const SearchTask task,
                               int max_n_bufs,
                               std::vector<std::vector<Node> >* node_list,
                               std::vector<std::vector<Edge> >* edge_list) {
  // extract features
  node_list->assign(states.size(), std::vector<Node>());
  edge_list->assign(states.size(), std::vector<Edge>());

  std::atomic<int> error_ct(0);

  ThreadPool& pool = ThreadPool::Global();
  pool.BeginBatch(static_cast<int>(states.size()) - 0);
  for (size_t i = 0; i < states.size(); ++i) {
    pool.Enqueue(GetGraph, states[i], task,
                 max_n_bufs, &(*node_list)[i], &(*edge_list)[i]);
  }
  pool.WaitBatch();

  if (error_ct > 0) {
    std::cerr << "Encountered " << error_ct
              << " errors during feature extraction. which are safely ignored." << std::endl;
  }
}

void GetGraphFromFile(const std::string& filename,
                             int n_lines,
                             int max_n_bufs,
                             std::vector<std::vector<Node> >* node_list,
                             std::vector<std::vector<Edge> >* edge_list,
                             std::vector<float>* normalized_throughputs,
                             std::vector<int>* task_ids) {
  Array<State> states;
  // ArrayNode* pstates = states.CopyOnWrite();
  std::vector<SearchTask> tasks;

  normalized_throughputs->clear();
  task_ids->clear();

  // (workload_key, target) -> (search_task, task_id)
  std::unordered_map<std::pair<std::string, std::string>, std::pair<SearchTask, size_t>> task_cache;
  // task_id -> min_cost
  std::vector<float> min_costs;

  // read from file
  RecordReader reader(filename);
  auto cur_inp = make_object<MeasureInputNode>();
  auto cur_res = make_object<MeasureResultNode>();
  while (reader->ReadNext(cur_inp.get(), cur_res.get())) {
    float cost = static_cast<float>(FloatArrayMean(cur_res->costs));
    const std::string& workload_key = cur_inp->task->workload_key;

    SearchTask task;
    size_t task_id;
    std::pair<std::string, std::string> key(workload_key, cur_inp->task->target->str());
    auto find_res = task_cache.find(key);
    if (find_res == task_cache.end()) {
      // rebuild task
      task = SearchTask(ComputeDAG(workload_key), workload_key,
                        cur_inp->task->target, cur_inp->task->target_host,
                        cur_inp->task->hardware_params);
      task_id = task_cache.size();

      // compute min cost for each task
      task_cache.insert(std::make_pair(key, std::make_pair(task, task_id)));
      min_costs.push_back(cost);
    } else {
      std::tie(task, task_id) = find_res->second;
      min_costs[task_id] = std::min(min_costs[task_id], cost);
    }

    tasks.push_back(std::move(task));
    task_ids->push_back(task_id);
    // pstates->data.push_back(cur_inp->state);
    states.push_back(cur_inp->state);
    normalized_throughputs->push_back(cost);

    if (n_lines > 0 && static_cast<int>(states.size()) >= n_lines) {
      break;
    }
  }

  for (size_t i = 0; i < normalized_throughputs->size(); ++i) {
    (*normalized_throughputs)[i] = min_costs[(*task_ids)[i]] / (*normalized_throughputs)[i];
  }

  GetGraphFromStates(states, tasks, max_n_bufs, node_list, edge_list);
}


void GetGraphFromMeasurePairs(const Array<MeasureInput>& inputs,
                                    const Array<MeasureResult>& results,
                                    int skip_first_n_feature_extraction,
                                    int max_n_bufs,
                                    std::vector<std::vector<Node> >* node_list,
                                    std::vector<std::vector<Edge> >* edge_list,
                                    std::vector<float>* normalized_throughputs,
                                    std::vector<int>* task_ids) {
  Array<State> states;
  // ArrayNode* pstates = states.CopyOnWrite();
  std::vector<SearchTask> tasks;

  normalized_throughputs->clear();
  task_ids->clear();

  // (workload_key, target) -> (search_task, task_id)
  std::unordered_map<std::pair<std::string, std::string>, std::pair<SearchTask, size_t>> task_cache;
  // task_id -> min_cost
  std::vector<float> min_costs;

  tasks.reserve(inputs.size());
  normalized_throughputs->reserve(inputs.size());
  task_ids->reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    float cost = static_cast<float>(FloatArrayMean(results[i]->costs));
    const std::string& workload_key = inputs[i]->task->workload_key;
    SearchTask task;

    size_t task_id;
    std::pair<std::string, std::string> key(workload_key, inputs[i]->task->target->str());
    auto find_res = task_cache.find(key);
    if (find_res == task_cache.end()) {
      if (inputs[i]->task->compute_dag.defined()) {   // the measure input is complete
          task = inputs[i]->task;
      } else {  // the measure input is incomplete
          // rebuild task for incomplete measure pairs read from file
          task = SearchTask(ComputeDAG(workload_key), workload_key,
                            inputs[i]->task->target, inputs[i]->task->target_host,
                            inputs[i]->task->hardware_params);
      }
      task_id = task_cache.size();

      // compute min cost for each task
      task_cache.insert(std::make_pair(key, std::make_pair(task, task_id)));
      min_costs.push_back(cost);
    } else {
      std::tie(task, task_id) = find_res->second;
      min_costs[task_id] = std::min(min_costs[task_id], cost);
    }

    tasks.push_back(std::move(task));
    task_ids->push_back(task_id);
    // pstates->data.push_back(inputs[i]->state);
    states.push_back(inputs[i]->state);
    normalized_throughputs->push_back(cost);
  }

  for (size_t i = 0; i < normalized_throughputs->size(); ++i) {
    (*normalized_throughputs)[i] = min_costs[(*task_ids)[i]] / (*normalized_throughputs)[i];
  }
  GetGraphFromStates(states, tasks, max_n_bufs, node_list, edge_list);
}



TVM_REGISTER_GLOBAL("ansor.GetGraphFromStates")
  .set_body([](TVMArgs args, TVMRetValue *ret) {

  Array<State> states = args[0];
  SearchTask task = args[1];
  int max_n_bufs = args[2];

  std::vector<std::vector<Node> > node_list;
  std::vector<std::vector<Edge> > edge_list;
  std::vector<float> normalized_throughputs;
  std::vector<int> task_ids;

  GetGraphFromStates(states, task, max_n_bufs, &node_list, &edge_list);
  std::vector<char> byte_data;
  *ret = SerializeGraph(edge_list, node_list, std::move(normalized_throughputs),
                        std::move(task_ids), &byte_data);

});

TVM_REGISTER_GLOBAL("ansor.GetGraphFromFile")
.set_body([](TVMArgs args, TVMRetValue *ret) {

  std::string filename = args[0];
  int n_lines = args[1];
  int max_n_bufs = args[2];

  std::vector<std::vector<Node> > node_list;
  std::vector<std::vector<Edge> > edge_list;
  std::vector<float> normalized_throughputs;
  std::vector<int> task_ids;


  GetGraphFromFile(filename, n_lines, max_n_bufs,
      &node_list, &edge_list, &normalized_throughputs, &task_ids);

  std::vector<char> byte_data;
  *ret = SerializeGraph(edge_list, node_list, std::move(normalized_throughputs),
                           std::move(task_ids), &byte_data);
});

TVM_REGISTER_GLOBAL("ansor.GetGraphFromMeasurePairs")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  Array<MeasureInput> inputs = args[0];
  Array<MeasureResult> results = args[1];
  int skip_first_n_feature_extraction = args[2];
  int max_n_bufs = args[3];

  std::vector<std::vector<Node> > node_list;
  std::vector<std::vector<Edge> > edge_list;
  std::vector<float> normalized_throughputs;
  std::vector<int> task_ids;

  GetGraphFromMeasurePairs(inputs, results, skip_first_n_feature_extraction, max_n_bufs,
      &node_list, &edge_list, &normalized_throughputs, &task_ids);

  std::vector<char> byte_data;
  *ret = SerializeGraph(edge_list, node_list, std::move(normalized_throughputs),
                           std::move(task_ids), &byte_data);
});


}   // namespace ansor
}   // namespace tvm
