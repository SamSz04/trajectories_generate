; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@global_smem = external addrspace(3) global [0 x i8], align 16
@shared_0 = private unnamed_addr addrspace(3) global [4096 x bfloat] undef

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #1

; Function Attrs: convergent nocallback nofree nounwind memory(argmem: read)
declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) readonly captures(none)) #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #3

; Function Attrs: nounwind
define ptx_kernel void @triton_softmax_5(ptr noalias align 16 dereferenceable(33554432) %arg0, ptr noalias align 256 dereferenceable(33554432) %arg1) local_unnamed_addr #4 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = zext i32 %3 to i64
  %5 = tail call range(i32 0, 128) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = shl i32 %5, 3
  %7 = zext i32 %6 to i64
  %8 = shl i64 %4, 12
  %9 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %8
  %10 = getelementptr [2 x i8], ptr addrspace(1) %9, i64 %7
  %11 = getelementptr i8, ptr addrspace(1) %10, i64 2048
  %12 = getelementptr i8, ptr addrspace(1) %10, i64 4096
  %13 = getelementptr i8, ptr addrspace(1) %10, i64 6144
  %14 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %10) #7
  %15 = extractvalue { i32, i32, i32, i32 } %14, 0
  %16 = bitcast i32 %15 to <2 x bfloat>
  %17 = extractvalue { i32, i32, i32, i32 } %14, 1
  %18 = bitcast i32 %17 to <2 x bfloat>
  %19 = extractvalue { i32, i32, i32, i32 } %14, 2
  %20 = bitcast i32 %19 to <2 x bfloat>
  %21 = extractvalue { i32, i32, i32, i32 } %14, 3
  %22 = bitcast i32 %21 to <2 x bfloat>
  %23 = extractelement <2 x bfloat> %16, i64 0
  %24 = extractelement <2 x bfloat> %16, i64 1
  %25 = extractelement <2 x bfloat> %18, i64 0
  %26 = extractelement <2 x bfloat> %18, i64 1
  %27 = extractelement <2 x bfloat> %20, i64 0
  %28 = extractelement <2 x bfloat> %20, i64 1
  %29 = extractelement <2 x bfloat> %22, i64 0
  %30 = extractelement <2 x bfloat> %22, i64 1
  %31 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %11) #7
  %32 = extractvalue { i32, i32, i32, i32 } %31, 0
  %33 = bitcast i32 %32 to <2 x bfloat>
  %34 = extractvalue { i32, i32, i32, i32 } %31, 1
  %35 = bitcast i32 %34 to <2 x bfloat>
  %36 = extractvalue { i32, i32, i32, i32 } %31, 2
  %37 = bitcast i32 %36 to <2 x bfloat>
  %38 = extractvalue { i32, i32, i32, i32 } %31, 3
  %39 = bitcast i32 %38 to <2 x bfloat>
  %40 = extractelement <2 x bfloat> %33, i64 0
  %41 = extractelement <2 x bfloat> %33, i64 1
  %42 = extractelement <2 x bfloat> %35, i64 0
  %43 = extractelement <2 x bfloat> %35, i64 1
  %44 = extractelement <2 x bfloat> %37, i64 0
  %45 = extractelement <2 x bfloat> %37, i64 1
  %46 = extractelement <2 x bfloat> %39, i64 0
  %47 = extractelement <2 x bfloat> %39, i64 1
  %48 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %12) #7
  %49 = extractvalue { i32, i32, i32, i32 } %48, 0
  %50 = bitcast i32 %49 to <2 x bfloat>
  %51 = extractvalue { i32, i32, i32, i32 } %48, 1
  %52 = bitcast i32 %51 to <2 x bfloat>
  %53 = extractvalue { i32, i32, i32, i32 } %48, 2
  %54 = bitcast i32 %53 to <2 x bfloat>
  %55 = extractvalue { i32, i32, i32, i32 } %48, 3
  %56 = bitcast i32 %55 to <2 x bfloat>
  %57 = extractelement <2 x bfloat> %50, i64 0
  %58 = extractelement <2 x bfloat> %50, i64 1
  %59 = extractelement <2 x bfloat> %52, i64 0
  %60 = extractelement <2 x bfloat> %52, i64 1
  %61 = extractelement <2 x bfloat> %54, i64 0
  %62 = extractelement <2 x bfloat> %54, i64 1
  %63 = extractelement <2 x bfloat> %56, i64 0
  %64 = extractelement <2 x bfloat> %56, i64 1
  %65 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %13) #7
  %66 = extractvalue { i32, i32, i32, i32 } %65, 0
  %67 = bitcast i32 %66 to <2 x bfloat>
  %68 = extractvalue { i32, i32, i32, i32 } %65, 1
  %69 = bitcast i32 %68 to <2 x bfloat>
  %70 = extractvalue { i32, i32, i32, i32 } %65, 2
  %71 = bitcast i32 %70 to <2 x bfloat>
  %72 = extractvalue { i32, i32, i32, i32 } %65, 3
  %73 = bitcast i32 %72 to <2 x bfloat>
  %74 = extractelement <2 x bfloat> %67, i64 0
  %75 = extractelement <2 x bfloat> %67, i64 1
  %76 = extractelement <2 x bfloat> %69, i64 0
  %77 = extractelement <2 x bfloat> %69, i64 1
  %78 = extractelement <2 x bfloat> %71, i64 0
  %79 = extractelement <2 x bfloat> %71, i64 1
  %80 = extractelement <2 x bfloat> %73, i64 0
  %81 = extractelement <2 x bfloat> %73, i64 1
  %82 = fpext bfloat %23 to float
  %83 = fpext bfloat %24 to float
  %84 = fpext bfloat %25 to float
  %85 = fpext bfloat %26 to float
  %86 = fpext bfloat %27 to float
  %87 = fpext bfloat %28 to float
  %88 = fpext bfloat %29 to float
  %89 = fpext bfloat %30 to float
  %90 = fpext bfloat %40 to float
  %91 = fpext bfloat %41 to float
  %92 = fpext bfloat %42 to float
  %93 = fpext bfloat %43 to float
  %94 = fpext bfloat %44 to float
  %95 = fpext bfloat %45 to float
  %96 = fpext bfloat %46 to float
  %97 = fpext bfloat %47 to float
  %98 = fpext bfloat %57 to float
  %99 = fpext bfloat %58 to float
  %100 = fpext bfloat %59 to float
  %101 = fpext bfloat %60 to float
  %102 = fpext bfloat %61 to float
  %103 = fpext bfloat %62 to float
  %104 = fpext bfloat %63 to float
  %105 = fpext bfloat %64 to float
  %106 = fpext bfloat %74 to float
  %107 = fpext bfloat %75 to float
  %108 = fpext bfloat %76 to float
  %109 = fpext bfloat %77 to float
  %110 = fpext bfloat %78 to float
  %111 = fpext bfloat %79 to float
  %112 = fpext bfloat %80 to float
  %113 = fpext bfloat %81 to float
  %114 = fmul float %82, %82
  %115 = fmul float %83, %83
  %116 = fmul float %84, %84
  %117 = fmul float %85, %85
  %118 = fmul float %86, %86
  %119 = fmul float %87, %87
  %120 = fmul float %88, %88
  %121 = fmul float %89, %89
  %122 = fmul float %90, %90
  %123 = fmul float %91, %91
  %124 = fmul float %92, %92
  %125 = fmul float %93, %93
  %126 = fmul float %94, %94
  %127 = fmul float %95, %95
  %128 = fmul float %96, %96
  %129 = fmul float %97, %97
  %130 = fmul float %98, %98
  %131 = fmul float %99, %99
  %132 = fmul float %100, %100
  %133 = fmul float %101, %101
  %134 = fmul float %102, %102
  %135 = fmul float %103, %103
  %136 = fmul float %104, %104
  %137 = fmul float %105, %105
  %138 = fmul float %106, %106
  %139 = fmul float %107, %107
  %140 = fmul float %108, %108
  %141 = fmul float %109, %109
  %142 = fmul float %110, %110
  %143 = fmul float %111, %111
  %144 = fmul float %112, %112
  %145 = fmul float %113, %113
  %146 = shl nuw nsw i32 %5, 4
  %147 = zext nneg i32 %146 to i64
  %148 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %147
  %149 = insertelement <4 x float> poison, float %114, i64 0
  %150 = insertelement <4 x float> %149, float %115, i64 1
  %151 = insertelement <4 x float> %150, float %116, i64 2
  %152 = insertelement <4 x float> %151, float %117, i64 3
  store <4 x float> %152, ptr addrspace(3) %148, align 16
  %153 = xor i32 %146, 2112
  %154 = zext nneg i32 %153 to i64
  %155 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %154
  %156 = insertelement <4 x float> poison, float %118, i64 0
  %157 = insertelement <4 x float> %156, float %119, i64 1
  %158 = insertelement <4 x float> %157, float %120, i64 2
  %159 = insertelement <4 x float> %158, float %121, i64 3
  store <4 x float> %159, ptr addrspace(3) %155, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %160 = shl nuw nsw i32 %5, 5
  %161 = and i32 %160, 768
  %162 = and i32 %6, 48
  %163 = and i32 %5, 96
  %164 = shl nuw nsw i32 %163, 1
  %165 = and i32 %5, 1
  %166 = icmp eq i32 %165, 0
  %167 = select i1 %166, i32 0, i32 2112
  %168 = or disjoint i32 %161, %162
  %169 = xor i32 %167, %164
  %170 = or disjoint i32 %168, %169
  %171 = zext nneg i32 %170 to i64
  %172 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %171
  %173 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %172)
  %174 = extractvalue { i32, i32, i32, i32 } %173, 0
  %175 = bitcast i32 %174 to float
  %176 = extractvalue { i32, i32, i32, i32 } %173, 1
  %177 = bitcast i32 %176 to float
  %178 = extractvalue { i32, i32, i32, i32 } %173, 2
  %179 = bitcast i32 %178 to float
  %180 = extractvalue { i32, i32, i32, i32 } %173, 3
  %181 = bitcast i32 %180 to float
  %182 = getelementptr inbounds nuw i8, ptr addrspace(3) %172, i64 1024
  %183 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %182)
  %184 = extractvalue { i32, i32, i32, i32 } %183, 0
  %185 = bitcast i32 %184 to float
  %186 = extractvalue { i32, i32, i32, i32 } %183, 1
  %187 = bitcast i32 %186 to float
  %188 = extractvalue { i32, i32, i32, i32 } %183, 2
  %189 = bitcast i32 %188 to float
  %190 = extractvalue { i32, i32, i32, i32 } %183, 3
  %191 = bitcast i32 %190 to float
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %192 = insertelement <4 x float> poison, float %122, i64 0
  %193 = insertelement <4 x float> %192, float %123, i64 1
  %194 = insertelement <4 x float> %193, float %124, i64 2
  %195 = insertelement <4 x float> %194, float %125, i64 3
  store <4 x float> %195, ptr addrspace(3) %148, align 16
  %196 = insertelement <4 x float> poison, float %126, i64 0
  %197 = insertelement <4 x float> %196, float %127, i64 1
  %198 = insertelement <4 x float> %197, float %128, i64 2
  %199 = insertelement <4 x float> %198, float %129, i64 3
  store <4 x float> %199, ptr addrspace(3) %155, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %200 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %172)
  %201 = extractvalue { i32, i32, i32, i32 } %200, 0
  %202 = bitcast i32 %201 to float
  %203 = extractvalue { i32, i32, i32, i32 } %200, 1
  %204 = bitcast i32 %203 to float
  %205 = extractvalue { i32, i32, i32, i32 } %200, 2
  %206 = bitcast i32 %205 to float
  %207 = extractvalue { i32, i32, i32, i32 } %200, 3
  %208 = bitcast i32 %207 to float
  %209 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %182)
  %210 = extractvalue { i32, i32, i32, i32 } %209, 0
  %211 = bitcast i32 %210 to float
  %212 = extractvalue { i32, i32, i32, i32 } %209, 1
  %213 = bitcast i32 %212 to float
  %214 = extractvalue { i32, i32, i32, i32 } %209, 2
  %215 = bitcast i32 %214 to float
  %216 = extractvalue { i32, i32, i32, i32 } %209, 3
  %217 = bitcast i32 %216 to float
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %218 = insertelement <4 x float> poison, float %130, i64 0
  %219 = insertelement <4 x float> %218, float %131, i64 1
  %220 = insertelement <4 x float> %219, float %132, i64 2
  %221 = insertelement <4 x float> %220, float %133, i64 3
  store <4 x float> %221, ptr addrspace(3) %148, align 16
  %222 = insertelement <4 x float> poison, float %134, i64 0
  %223 = insertelement <4 x float> %222, float %135, i64 1
  %224 = insertelement <4 x float> %223, float %136, i64 2
  %225 = insertelement <4 x float> %224, float %137, i64 3
  store <4 x float> %225, ptr addrspace(3) %155, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %226 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %172)
  %227 = extractvalue { i32, i32, i32, i32 } %226, 0
  %228 = bitcast i32 %227 to float
  %229 = extractvalue { i32, i32, i32, i32 } %226, 1
  %230 = bitcast i32 %229 to float
  %231 = extractvalue { i32, i32, i32, i32 } %226, 2
  %232 = bitcast i32 %231 to float
  %233 = extractvalue { i32, i32, i32, i32 } %226, 3
  %234 = bitcast i32 %233 to float
  %235 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %182)
  %236 = extractvalue { i32, i32, i32, i32 } %235, 0
  %237 = bitcast i32 %236 to float
  %238 = extractvalue { i32, i32, i32, i32 } %235, 1
  %239 = bitcast i32 %238 to float
  %240 = extractvalue { i32, i32, i32, i32 } %235, 2
  %241 = bitcast i32 %240 to float
  %242 = extractvalue { i32, i32, i32, i32 } %235, 3
  %243 = bitcast i32 %242 to float
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %244 = insertelement <4 x float> poison, float %138, i64 0
  %245 = insertelement <4 x float> %244, float %139, i64 1
  %246 = insertelement <4 x float> %245, float %140, i64 2
  %247 = insertelement <4 x float> %246, float %141, i64 3
  store <4 x float> %247, ptr addrspace(3) %148, align 16
  %248 = insertelement <4 x float> poison, float %142, i64 0
  %249 = insertelement <4 x float> %248, float %143, i64 1
  %250 = insertelement <4 x float> %249, float %144, i64 2
  %251 = insertelement <4 x float> %250, float %145, i64 3
  store <4 x float> %251, ptr addrspace(3) %155, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %252 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %172)
  %253 = extractvalue { i32, i32, i32, i32 } %252, 0
  %254 = bitcast i32 %253 to float
  %255 = extractvalue { i32, i32, i32, i32 } %252, 1
  %256 = bitcast i32 %255 to float
  %257 = extractvalue { i32, i32, i32, i32 } %252, 2
  %258 = bitcast i32 %257 to float
  %259 = extractvalue { i32, i32, i32, i32 } %252, 3
  %260 = bitcast i32 %259 to float
  %261 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %182)
  %262 = extractvalue { i32, i32, i32, i32 } %261, 0
  %263 = bitcast i32 %262 to float
  %264 = extractvalue { i32, i32, i32, i32 } %261, 1
  %265 = bitcast i32 %264 to float
  %266 = extractvalue { i32, i32, i32, i32 } %261, 2
  %267 = bitcast i32 %266 to float
  %268 = extractvalue { i32, i32, i32, i32 } %261, 3
  %269 = bitcast i32 %268 to float
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %270 = fadd float %175, %177
  %271 = fadd float %179, %181
  %272 = fadd float %185, %187
  %273 = fadd float %189, %191
  %274 = fadd float %202, %204
  %275 = fadd float %206, %208
  %276 = fadd float %211, %213
  %277 = fadd float %215, %217
  %278 = fadd float %228, %230
  %279 = fadd float %232, %234
  %280 = fadd float %237, %239
  %281 = fadd float %241, %243
  %282 = fadd float %254, %256
  %283 = fadd float %258, %260
  %284 = fadd float %263, %265
  %285 = fadd float %267, %269
  %286 = fadd float %270, %271
  %287 = fadd float %272, %273
  %288 = fadd float %274, %275
  %289 = fadd float %276, %277
  %290 = fadd float %278, %279
  %291 = fadd float %280, %281
  %292 = fadd float %282, %283
  %293 = fadd float %284, %285
  %294 = fadd float %286, %287
  %295 = fadd float %288, %289
  %296 = fadd float %290, %291
  %297 = fadd float %292, %293
  %298 = fadd float %294, %295
  %299 = fadd float %296, %297
  %300 = fadd float %298, %299
  %301 = bitcast float %300 to i32
  %302 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %301, i32 16, i32 31)
  %303 = bitcast i32 %302 to float
  %304 = fadd float %300, %303
  %305 = bitcast float %304 to i32
  %306 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %305, i32 8, i32 31)
  %307 = bitcast i32 %306 to float
  %308 = fadd float %304, %307
  %309 = bitcast float %308 to i32
  %310 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %309, i32 4, i32 31)
  %311 = bitcast i32 %310 to float
  %312 = fadd float %308, %311
  %313 = bitcast float %312 to i32
  %314 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %313, i32 2, i32 31)
  %315 = bitcast i32 %314 to float
  %316 = fadd float %312, %315
  %317 = bitcast float %316 to i32
  %318 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %317, i32 1, i32 31)
  %319 = bitcast i32 %318 to float
  %320 = fadd float %316, %319
  %321 = lshr exact i32 %163, 3
  %322 = zext nneg i32 %321 to i64
  %323 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %322
  store float %320, ptr addrspace(3) %323, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %324 = shl nuw nsw i32 %5, 2
  %325 = and i32 %324, 12
  %326 = zext nneg i32 %325 to i64
  %327 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %326
  %328 = load i32, ptr addrspace(3) %327, align 4
  %329 = bitcast i32 %328 to float
  %330 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %328, i32 2, i32 31)
  %331 = bitcast i32 %330 to float
  %332 = fadd float %329, %331
  %333 = bitcast float %332 to i32
  %334 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %333, i32 1, i32 31)
  %335 = bitcast i32 %334 to float
  %336 = fadd float %332, %335
  %337 = fmul float %336, 0x3F30000000000000
  %338 = fadd float %337, 0x3EE5000000000000
  %339 = tail call float @llvm.nvvm.rsqrt.approx.f(float %338)
  %340 = fptrunc float %339 to bfloat
  %341 = fmul bfloat %23, %340
  %342 = fmul bfloat %24, %340
  %343 = fmul bfloat %25, %340
  %344 = fmul bfloat %26, %340
  %345 = fmul bfloat %27, %340
  %346 = fmul bfloat %28, %340
  %347 = fmul bfloat %29, %340
  %348 = fmul bfloat %30, %340
  %349 = fmul bfloat %40, %340
  %350 = fmul bfloat %41, %340
  %351 = fmul bfloat %42, %340
  %352 = fmul bfloat %43, %340
  %353 = fmul bfloat %44, %340
  %354 = fmul bfloat %45, %340
  %355 = fmul bfloat %46, %340
  %356 = fmul bfloat %47, %340
  %357 = fmul bfloat %57, %340
  %358 = fmul bfloat %58, %340
  %359 = fmul bfloat %59, %340
  %360 = fmul bfloat %60, %340
  %361 = fmul bfloat %61, %340
  %362 = fmul bfloat %62, %340
  %363 = fmul bfloat %63, %340
  %364 = fmul bfloat %64, %340
  %365 = fmul bfloat %74, %340
  %366 = fmul bfloat %75, %340
  %367 = fmul bfloat %76, %340
  %368 = fmul bfloat %77, %340
  %369 = fmul bfloat %78, %340
  %370 = fmul bfloat %79, %340
  %371 = fmul bfloat %80, %340
  %372 = fmul bfloat %81, %340
  %373 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %8
  %374 = getelementptr [2 x i8], ptr addrspace(1) %373, i64 %7
  %375 = getelementptr i8, ptr addrspace(1) %374, i64 2048
  %376 = getelementptr i8, ptr addrspace(1) %374, i64 4096
  %377 = getelementptr i8, ptr addrspace(1) %374, i64 6144
  %378 = insertelement <2 x bfloat> poison, bfloat %341, i64 0
  %379 = insertelement <2 x bfloat> %378, bfloat %342, i64 1
  %380 = bitcast <2 x bfloat> %379 to i32
  %381 = insertelement <2 x bfloat> poison, bfloat %343, i64 0
  %382 = insertelement <2 x bfloat> %381, bfloat %344, i64 1
  %383 = bitcast <2 x bfloat> %382 to i32
  %384 = insertelement <2 x bfloat> poison, bfloat %345, i64 0
  %385 = insertelement <2 x bfloat> %384, bfloat %346, i64 1
  %386 = bitcast <2 x bfloat> %385 to i32
  %387 = insertelement <2 x bfloat> poison, bfloat %347, i64 0
  %388 = insertelement <2 x bfloat> %387, bfloat %348, i64 1
  %389 = bitcast <2 x bfloat> %388 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %380, i32 %383, i32 %386, i32 %389, ptr addrspace(1) %374) #7
  %390 = insertelement <2 x bfloat> poison, bfloat %349, i64 0
  %391 = insertelement <2 x bfloat> %390, bfloat %350, i64 1
  %392 = bitcast <2 x bfloat> %391 to i32
  %393 = insertelement <2 x bfloat> poison, bfloat %351, i64 0
  %394 = insertelement <2 x bfloat> %393, bfloat %352, i64 1
  %395 = bitcast <2 x bfloat> %394 to i32
  %396 = insertelement <2 x bfloat> poison, bfloat %353, i64 0
  %397 = insertelement <2 x bfloat> %396, bfloat %354, i64 1
  %398 = bitcast <2 x bfloat> %397 to i32
  %399 = insertelement <2 x bfloat> poison, bfloat %355, i64 0
  %400 = insertelement <2 x bfloat> %399, bfloat %356, i64 1
  %401 = bitcast <2 x bfloat> %400 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %392, i32 %395, i32 %398, i32 %401, ptr addrspace(1) %375) #7
  %402 = insertelement <2 x bfloat> poison, bfloat %357, i64 0
  %403 = insertelement <2 x bfloat> %402, bfloat %358, i64 1
  %404 = bitcast <2 x bfloat> %403 to i32
  %405 = insertelement <2 x bfloat> poison, bfloat %359, i64 0
  %406 = insertelement <2 x bfloat> %405, bfloat %360, i64 1
  %407 = bitcast <2 x bfloat> %406 to i32
  %408 = insertelement <2 x bfloat> poison, bfloat %361, i64 0
  %409 = insertelement <2 x bfloat> %408, bfloat %362, i64 1
  %410 = bitcast <2 x bfloat> %409 to i32
  %411 = insertelement <2 x bfloat> poison, bfloat %363, i64 0
  %412 = insertelement <2 x bfloat> %411, bfloat %364, i64 1
  %413 = bitcast <2 x bfloat> %412 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %404, i32 %407, i32 %410, i32 %413, ptr addrspace(1) %376) #7
  %414 = insertelement <2 x bfloat> poison, bfloat %365, i64 0
  %415 = insertelement <2 x bfloat> %414, bfloat %366, i64 1
  %416 = bitcast <2 x bfloat> %415 to i32
  %417 = insertelement <2 x bfloat> poison, bfloat %367, i64 0
  %418 = insertelement <2 x bfloat> %417, bfloat %368, i64 1
  %419 = bitcast <2 x bfloat> %418 to i32
  %420 = insertelement <2 x bfloat> poison, bfloat %369, i64 0
  %421 = insertelement <2 x bfloat> %420, bfloat %370, i64 1
  %422 = bitcast <2 x bfloat> %421 to i32
  %423 = insertelement <2 x bfloat> poison, bfloat %371, i64 0
  %424 = insertelement <2 x bfloat> %423, bfloat %372, i64 1
  %425 = bitcast <2 x bfloat> %424 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %416, i32 %419, i32 %422, i32 %425, ptr addrspace(1) %377) #7
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_convert_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(67108864) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %10
  %12 = load <4 x bfloat>, ptr addrspace(1) %11, align 8, !invariant.load !6
  %13 = extractelement <4 x bfloat> %12, i32 0
  %14 = extractelement <4 x bfloat> %12, i32 1
  %15 = extractelement <4 x bfloat> %12, i32 2
  %16 = extractelement <4 x bfloat> %12, i32 3
  %17 = fpext bfloat %13 to float
  %18 = fpext bfloat %14 to float
  %19 = fpext bfloat %15 to float
  %20 = fpext bfloat %16 to float
  %21 = insertelement <4 x float> poison, float %17, i64 0
  %22 = insertelement <4 x float> %21, float %18, i64 1
  %23 = insertelement <4 x float> %22, float %19, i64 2
  %24 = insertelement <4 x float> %23, float %20, i64 3
  %25 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %10
  store <4 x float> %24, ptr addrspace(1) %25, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_convert_1(ptr noalias readonly align 16 captures(none) dereferenceable(8192) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(16384) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !7
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %5, 7
  %8 = or disjoint i32 %7, %6
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %9
  %11 = load bfloat, ptr addrspace(1) %10, align 2, !invariant.load !6
  %12 = fpext bfloat %11 to float
  %13 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %9
  store float %12, ptr addrspace(1) %13, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_broadcast_1(ptr noalias readonly align 256 captures(none) dereferenceable(16384) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(67108864) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = and i32 %8, 3584
  %11 = or disjoint i32 %10, %7
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds [4 x i8], ptr addrspace(1) %3, i64 %12
  %14 = load <4 x float>, ptr addrspace(1) %13, align 16, !invariant.load !6
  %15 = zext nneg i32 %9 to i64
  %16 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %15
  store <4 x float> %14, ptr addrspace(1) %16, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_multiply(ptr noalias align 256 captures(none) dereferenceable(67108864) %0, ptr noalias readonly align 256 captures(none) dereferenceable(67108864) %1, ptr noalias readnone align 256 captures(none) dereferenceable(67108864) %2) local_unnamed_addr #5 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %8 = shl nuw nsw i32 %7, 2
  %9 = shl nuw nsw i32 %6, 9
  %10 = or disjoint i32 %8, %9
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %11
  %13 = getelementptr inbounds [4 x i8], ptr addrspace(1) %5, i64 %11
  %14 = load <4 x float>, ptr addrspace(1) %13, align 16
  %15 = extractelement <4 x float> %14, i32 0
  %16 = extractelement <4 x float> %14, i32 1
  %17 = extractelement <4 x float> %14, i32 2
  %18 = extractelement <4 x float> %14, i32 3
  %19 = load <4 x float>, ptr addrspace(1) %12, align 16, !invariant.load !6
  %20 = extractelement <4 x float> %19, i32 0
  %21 = extractelement <4 x float> %19, i32 1
  %22 = extractelement <4 x float> %19, i32 2
  %23 = extractelement <4 x float> %19, i32 3
  %24 = fmul float %20, %15
  %25 = fmul float %21, %16
  %26 = fmul float %22, %17
  %27 = fmul float %23, %18
  %28 = insertelement <4 x float> poison, float %24, i64 0
  %29 = insertelement <4 x float> %28, float %25, i64 1
  %30 = insertelement <4 x float> %29, float %26, i64 2
  %31 = insertelement <4 x float> %30, float %27, i64 3
  store <4 x float> %31, ptr addrspace(1) %13, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_convert_2(ptr noalias readonly align 256 captures(none) dereferenceable(67108864) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(33554432) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds [4 x i8], ptr addrspace(1) %3, i64 %10
  %12 = load <4 x float>, ptr addrspace(1) %11, align 16, !invariant.load !6
  %13 = extractelement <4 x float> %12, i32 0
  %14 = extractelement <4 x float> %12, i32 1
  %15 = extractelement <4 x float> %12, i32 2
  %16 = extractelement <4 x float> %12, i32 3
  %17 = fptrunc float %13 to bfloat
  %18 = fptrunc float %14 to bfloat
  %19 = fptrunc float %15 to bfloat
  %20 = fptrunc float %16 to bfloat
  %21 = insertelement <4 x bfloat> poison, bfloat %17, i64 0
  %22 = insertelement <4 x bfloat> %21, bfloat %18, i64 1
  %23 = insertelement <4 x bfloat> %22, bfloat %19, i64 2
  %24 = insertelement <4 x bfloat> %23, bfloat %20, i64 3
  %25 = getelementptr inbounds [2 x i8], ptr addrspace(1) %4, i64 %10
  store <4 x bfloat> %24, ptr addrspace(1) %25, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @input_concatenate_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(8388608) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8388608) %1, ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(50331648) %3) local_unnamed_addr #5 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %3 to ptr addrspace(1)
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = addrspacecast ptr %0 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %11 = and i32 %9, 7
  %12 = shl nuw nsw i32 %11, 9
  %13 = lshr i32 %9, 3
  %14 = mul nuw nsw i32 %13, 6144
  %15 = add nuw nsw i32 %12, %14
  %16 = shl nuw nsw i32 %10, 2
  %17 = or disjoint i32 %15, %16
  %18 = shl nuw nsw i32 %9, 9
  %19 = or disjoint i32 %18, %16
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds [2 x i8], ptr addrspace(1) %5, i64 %20
  %22 = load <4 x bfloat>, ptr addrspace(1) %21, align 8, !invariant.load !6
  %23 = zext nneg i32 %17 to i64
  %24 = getelementptr inbounds [2 x i8], ptr addrspace(1) %6, i64 %23
  store <4 x bfloat> %22, ptr addrspace(1) %24, align 8
  %25 = icmp samesign ult i32 %11, 2
  br i1 %25, label %.critedge, label %.critedge6

.critedge:                                        ; preds = %4
  %26 = shl nuw nsw i32 %13, 10
  %27 = or disjoint i32 %12, %26
  %28 = or disjoint i32 %27, %16
  %29 = zext nneg i32 %28 to i64
  %30 = getelementptr inbounds [2 x i8], ptr addrspace(1) %7, i64 %29
  %31 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 8192
  %32 = load <4 x bfloat>, ptr addrspace(1) %30, align 8, !invariant.load !6
  %33 = shufflevector <4 x bfloat> %32, <4 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %34 = shufflevector <4 x bfloat> %32, <4 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %35 = extractelement <2 x bfloat> %33, i32 0
  %36 = insertelement <4 x bfloat> poison, bfloat %35, i32 0
  %37 = extractelement <2 x bfloat> %33, i32 1
  %38 = insertelement <4 x bfloat> %36, bfloat %37, i32 1
  %39 = extractelement <2 x bfloat> %34, i32 0
  %40 = insertelement <4 x bfloat> %38, bfloat %39, i32 2
  %41 = extractelement <2 x bfloat> %34, i32 1
  %42 = insertelement <4 x bfloat> %40, bfloat %41, i32 3
  store <4 x bfloat> %42, ptr addrspace(1) %31, align 8
  %43 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 10240
  %44 = getelementptr inbounds [2 x i8], ptr addrspace(1) %8, i64 %29
  %45 = load <4 x bfloat>, ptr addrspace(1) %44, align 8, !invariant.load !6
  %46 = shufflevector <4 x bfloat> %45, <4 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %47 = shufflevector <4 x bfloat> %45, <4 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %48 = extractelement <2 x bfloat> %46, i32 0
  %49 = insertelement <4 x bfloat> poison, bfloat %48, i32 0
  %50 = extractelement <2 x bfloat> %46, i32 1
  %51 = insertelement <4 x bfloat> %49, bfloat %50, i32 1
  %52 = extractelement <2 x bfloat> %47, i32 0
  %53 = insertelement <4 x bfloat> %51, bfloat %52, i32 2
  %54 = extractelement <2 x bfloat> %47, i32 1
  %55 = insertelement <4 x bfloat> %53, bfloat %54, i32 3
  store <4 x bfloat> %55, ptr addrspace(1) %43, align 8
  br label %.critedge6

.critedge6:                                       ; preds = %4, %.critedge
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_slice_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(50331648) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8388608) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(8388608) %2) local_unnamed_addr #5 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !8
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %9 = shl nuw nsw i32 %8, 2
  %10 = shl nuw nsw i32 %7, 9
  %11 = or disjoint i32 %9, %10
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds [2 x i8], ptr addrspace(1) %4, i64 %12
  %14 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 33554432
  %15 = load <4 x bfloat>, ptr addrspace(1) %14, align 8, !invariant.load !6
  %16 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 41943040
  %17 = load <4 x bfloat>, ptr addrspace(1) %16, align 8, !invariant.load !6
  %18 = getelementptr inbounds [2 x i8], ptr addrspace(1) %5, i64 %12
  store <4 x bfloat> %17, ptr addrspace(1) %18, align 8
  %19 = getelementptr inbounds [2 x i8], ptr addrspace(1) %6, i64 %12
  store <4 x bfloat> %15, ptr addrspace(1) %19, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_broadcast_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8388608) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(33554432) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = and i32 %8, 523776
  %11 = shl nuw nsw i32 %5, 7
  %12 = and i32 %11, 3670016
  %13 = or disjoint i32 %12, %10
  %14 = or disjoint i32 %13, %7
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %15
  %17 = load <4 x bfloat>, ptr addrspace(1) %16, align 8, !invariant.load !6
  %18 = zext nneg i32 %9 to i64
  %19 = getelementptr inbounds [2 x i8], ptr addrspace(1) %4, i64 %18
  store <4 x bfloat> %17, ptr addrspace(1) %19, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_slice_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(50331648) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(33554432) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %10
  %12 = load <4 x bfloat>, ptr addrspace(1) %11, align 8, !invariant.load !6
  %13 = getelementptr inbounds [2 x i8], ptr addrspace(1) %4, i64 %10
  store <4 x bfloat> %12, ptr addrspace(1) %13, align 8
  ret void
}

; Function Attrs: nounwind
define ptx_kernel void @gemm_fusion_dot_2(ptr noalias align 256 dereferenceable(33554432) %arg0, ptr noalias align 256 dereferenceable(33554432) %arg1, ptr noalias align 256 dereferenceable(1073741824) %arg2) local_unnamed_addr #6 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = addrspacecast ptr %arg2 to ptr addrspace(1)
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = zext i32 %4 to i64
  %6 = lshr i64 %5, 11
  %7 = lshr i64 %5, 5
  %8 = and i64 %7, 63
  %9 = tail call range(i32 0, 64) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %10 = and i32 %9, 7
  %11 = shl i32 %10, 3
  %12 = and i32 %9, 48
  %13 = shl i32 %9, 9
  %14 = and i32 %13, 28672
  %15 = or disjoint i32 %14, %11
  %16 = zext nneg i32 %15 to i64
  %17 = shl nuw nsw i64 %6, 19
  %18 = shl nuw nsw i64 %5, 7
  %19 = and i64 %18, 3968
  %20 = shl i32 %9, 8
  %21 = and i32 %20, 12288
  %22 = and i32 %9, 15
  %23 = shl i32 %22, 3
  %24 = or disjoint i32 %23, %21
  %25 = zext nneg i32 %24 to i64
  %.idx319 = shl i64 %8, 7
  %26 = getelementptr i8, ptr addrspace(1) %1, i64 %.idx319
  %27 = getelementptr [2 x i8], ptr addrspace(1) %26, i64 %17
  %28 = getelementptr [2 x i8], ptr addrspace(1) %27, i64 %16
  %29 = getelementptr i8, ptr addrspace(1) %28, i64 65536
  %30 = shl nuw nsw i32 %9, 4
  %31 = shl nuw nsw i32 %9, 1
  %32 = and i32 %31, 112
  %33 = xor i32 %32, %30
  %34 = zext i32 %33 to i64
  %35 = getelementptr i8, ptr addrspace(3) @global_smem, i64 %34
  %36 = getelementptr i8, ptr addrspace(3) %35, i64 12288
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) %36, ptr addrspace(1) %28) #7
  %37 = getelementptr i8, ptr addrspace(3) %35, i64 13312
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) nonnull %37, ptr addrspace(1) %29) #7
  tail call void @llvm.nvvm.cp.async.commit.group()
  %38 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %19
  %39 = getelementptr [2 x i8], ptr addrspace(1) %38, i64 %17
  %40 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %25
  %41 = getelementptr i8, ptr addrspace(1) %40, i64 32768
  %42 = getelementptr i8, ptr addrspace(1) %40, i64 65536
  %43 = getelementptr i8, ptr addrspace(1) %40, i64 98304
  %44 = xor i32 %30, %12
  %45 = zext i32 %44 to i64
  %46 = getelementptr i8, ptr addrspace(3) @global_smem, i64 %45
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) %46, ptr addrspace(1) %40) #7
  %47 = getelementptr inbounds nuw i8, ptr addrspace(3) %46, i64 2048
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) nonnull %47, ptr addrspace(1) %42) #7
  %48 = xor i32 %44, 1088
  %49 = zext i32 %48 to i64
  %50 = getelementptr i8, ptr addrspace(3) @global_smem, i64 %49
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) %50, ptr addrspace(1) %41) #7
  %51 = getelementptr inbounds nuw i8, ptr addrspace(3) %50, i64 2048
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) nonnull %51, ptr addrspace(1) %43) #7
  tail call void @llvm.nvvm.cp.async.commit.group()
  %52 = getelementptr i8, ptr addrspace(1) %28, i64 131072
  %53 = getelementptr i8, ptr addrspace(1) %28, i64 196608
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %54 = getelementptr i8, ptr addrspace(3) %35, i64 14336
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) %54, ptr addrspace(1) %52) #7
  %55 = getelementptr i8, ptr addrspace(3) %35, i64 15360
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) nonnull %55, ptr addrspace(1) %53) #7
  tail call void @llvm.nvvm.cp.async.commit.group()
  %56 = getelementptr i8, ptr addrspace(1) %40, i64 131072
  %57 = getelementptr i8, ptr addrspace(1) %40, i64 163840
  %58 = getelementptr i8, ptr addrspace(1) %40, i64 196608
  %59 = getelementptr i8, ptr addrspace(1) %40, i64 229376
  %60 = getelementptr i8, ptr addrspace(3) %46, i64 4096
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) %60, ptr addrspace(1) %56) #7
  %61 = getelementptr i8, ptr addrspace(3) %46, i64 6144
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) nonnull %61, ptr addrspace(1) %58) #7
  %62 = getelementptr i8, ptr addrspace(3) %50, i64 4096
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) %62, ptr addrspace(1) %57) #7
  %63 = getelementptr i8, ptr addrspace(3) %50, i64 6144
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) nonnull %63, ptr addrspace(1) %59) #7
  tail call void @llvm.nvvm.cp.async.commit.group()
  %64 = getelementptr i8, ptr addrspace(1) %28, i64 262144
  %65 = getelementptr i8, ptr addrspace(1) %28, i64 327680
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %66 = getelementptr i8, ptr addrspace(3) %35, i64 16384
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) %66, ptr addrspace(1) %64) #7
  %67 = getelementptr i8, ptr addrspace(3) %35, i64 17408
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) nonnull %67, ptr addrspace(1) %65) #7
  tail call void @llvm.nvvm.cp.async.commit.group()
  %68 = getelementptr i8, ptr addrspace(1) %40, i64 262144
  %69 = getelementptr i8, ptr addrspace(1) %40, i64 294912
  %70 = getelementptr i8, ptr addrspace(1) %40, i64 327680
  %71 = getelementptr i8, ptr addrspace(1) %40, i64 360448
  %72 = getelementptr i8, ptr addrspace(3) %46, i64 8192
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) %72, ptr addrspace(1) %68) #7
  %73 = getelementptr i8, ptr addrspace(3) %46, i64 10240
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) nonnull %73, ptr addrspace(1) %70) #7
  %74 = getelementptr i8, ptr addrspace(3) %50, i64 8192
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) %74, ptr addrspace(1) %69) #7
  %75 = getelementptr i8, ptr addrspace(3) %50, i64 10240
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, 0x10;", "r,l"(ptr addrspace(3) nonnull %75, ptr addrspace(1) %71) #7
  tail call void @llvm.nvvm.cp.async.commit.group()
  tail call void @llvm.nvvm.cp.async.wait.group(i32 4)
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %76 = shl nuw nsw i32 %10, 7
  %77 = shl nuw nsw i32 %10, 4
  %78 = and i32 %9, 8
  %79 = shl nuw nsw i32 %78, 1
  %80 = and i32 %9, 16
  %81 = shl nuw nsw i32 %80, 6
  %82 = or disjoint i32 %76, %81
  %83 = xor i32 %77, %79
  %84 = or disjoint i32 %82, %83
  %85 = zext nneg i32 %84 to i64
  %86 = getelementptr i8, ptr addrspace(3) @global_smem, i64 %85
  %87 = getelementptr i8, ptr addrspace(3) %86, i64 12288
  %88 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %87)
  %89 = extractvalue { i32, i32, i32, i32 } %88, 0
  %90 = bitcast i32 %89 to <2 x bfloat>
  %91 = extractelement <2 x bfloat> %90, i64 0
  %92 = extractelement <2 x bfloat> %90, i64 1
  %93 = extractvalue { i32, i32, i32, i32 } %88, 1
  %94 = bitcast i32 %93 to <2 x bfloat>
  %95 = extractelement <2 x bfloat> %94, i64 0
  %96 = extractelement <2 x bfloat> %94, i64 1
  %97 = extractvalue { i32, i32, i32, i32 } %88, 2
  %98 = bitcast i32 %97 to <2 x bfloat>
  %99 = extractelement <2 x bfloat> %98, i64 0
  %100 = extractelement <2 x bfloat> %98, i64 1
  %101 = extractvalue { i32, i32, i32, i32 } %88, 3
  %102 = bitcast i32 %101 to <2 x bfloat>
  %103 = extractelement <2 x bfloat> %102, i64 0
  %104 = extractelement <2 x bfloat> %102, i64 1
  %105 = xor i32 %84, 32
  %106 = zext nneg i32 %105 to i64
  %107 = getelementptr i8, ptr addrspace(3) @global_smem, i64 %106
  %108 = getelementptr i8, ptr addrspace(3) %107, i64 12288
  %109 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %108)
  %110 = extractvalue { i32, i32, i32, i32 } %109, 0
  %111 = bitcast i32 %110 to <2 x bfloat>
  %112 = extractelement <2 x bfloat> %111, i64 0
  %113 = extractelement <2 x bfloat> %111, i64 1
  %114 = extractvalue { i32, i32, i32, i32 } %109, 1
  %115 = bitcast i32 %114 to <2 x bfloat>
  %116 = extractelement <2 x bfloat> %115, i64 0
  %117 = extractelement <2 x bfloat> %115, i64 1
  %118 = extractvalue { i32, i32, i32, i32 } %109, 2
  %119 = bitcast i32 %118 to <2 x bfloat>
  %120 = extractelement <2 x bfloat> %119, i64 0
  %121 = extractelement <2 x bfloat> %119, i64 1
  %122 = extractvalue { i32, i32, i32, i32 } %109, 3
  %123 = bitcast i32 %122 to <2 x bfloat>
  %124 = extractelement <2 x bfloat> %123, i64 0
  %125 = extractelement <2 x bfloat> %123, i64 1
  %126 = xor i32 %84, 64
  %127 = zext nneg i32 %126 to i64
  %128 = getelementptr i8, ptr addrspace(3) @global_smem, i64 %127
  %129 = getelementptr i8, ptr addrspace(3) %128, i64 12288
  %130 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %129)
  %131 = extractvalue { i32, i32, i32, i32 } %130, 0
  %132 = bitcast i32 %131 to <2 x bfloat>
  %133 = extractelement <2 x bfloat> %132, i64 0
  %134 = extractelement <2 x bfloat> %132, i64 1
  %135 = extractvalue { i32, i32, i32, i32 } %130, 1
  %136 = bitcast i32 %135 to <2 x bfloat>
  %137 = extractelement <2 x bfloat> %136, i64 0
  %138 = extractelement <2 x bfloat> %136, i64 1
  %139 = extractvalue { i32, i32, i32, i32 } %130, 2
  %140 = bitcast i32 %139 to <2 x bfloat>
  %141 = extractelement <2 x bfloat> %140, i64 0
  %142 = extractelement <2 x bfloat> %140, i64 1
  %143 = extractvalue { i32, i32, i32, i32 } %130, 3
  %144 = bitcast i32 %143 to <2 x bfloat>
  %145 = extractelement <2 x bfloat> %144, i64 0
  %146 = extractelement <2 x bfloat> %144, i64 1
  %147 = xor i32 %84, 96
  %148 = zext nneg i32 %147 to i64
  %149 = getelementptr i8, ptr addrspace(3) @global_smem, i64 %148
  %150 = getelementptr i8, ptr addrspace(3) %149, i64 12288
  %151 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %150)
  %152 = extractvalue { i32, i32, i32, i32 } %151, 0
  %153 = bitcast i32 %152 to <2 x bfloat>
  %154 = extractelement <2 x bfloat> %153, i64 0
  %155 = extractelement <2 x bfloat> %153, i64 1
  %156 = extractvalue { i32, i32, i32, i32 } %151, 1
  %157 = bitcast i32 %156 to <2 x bfloat>
  %158 = extractelement <2 x bfloat> %157, i64 0
  %159 = extractelement <2 x bfloat> %157, i64 1
  %160 = extractvalue { i32, i32, i32, i32 } %151, 2
  %161 = bitcast i32 %160 to <2 x bfloat>
  %162 = extractelement <2 x bfloat> %161, i64 0
  %163 = extractelement <2 x bfloat> %161, i64 1
  %164 = extractvalue { i32, i32, i32, i32 } %151, 3
  %165 = bitcast i32 %164 to <2 x bfloat>
  %166 = extractelement <2 x bfloat> %165, i64 0
  %167 = extractelement <2 x bfloat> %165, i64 1
  %168 = shl nuw nsw i32 %22, 8
  %169 = shl nuw nsw i32 %80, 1
  %170 = and i32 %9, 32
  %171 = lshr exact i32 %170, 1
  %172 = xor i32 %77, %169
  %173 = xor i32 %172, %171
  %174 = or disjoint i32 %173, %168
  %175 = zext nneg i32 %174 to i64
  %176 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %175
  %177 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %176)
  %178 = extractvalue { i32, i32, i32, i32 } %177, 0
  %179 = bitcast i32 %178 to <2 x bfloat>
  %180 = extractelement <2 x bfloat> %179, i64 0
  %181 = extractelement <2 x bfloat> %179, i64 1
  %182 = extractvalue { i32, i32, i32, i32 } %177, 1
  %183 = bitcast i32 %182 to <2 x bfloat>
  %184 = extractelement <2 x bfloat> %183, i64 0
  %185 = extractelement <2 x bfloat> %183, i64 1
  %186 = extractvalue { i32, i32, i32, i32 } %177, 2
  %187 = bitcast i32 %186 to <2 x bfloat>
  %188 = extractelement <2 x bfloat> %187, i64 0
  %189 = extractelement <2 x bfloat> %187, i64 1
  %190 = extractvalue { i32, i32, i32, i32 } %177, 3
  %191 = bitcast i32 %190 to <2 x bfloat>
  %192 = extractelement <2 x bfloat> %191, i64 0
  %193 = extractelement <2 x bfloat> %191, i64 1
  %194 = getelementptr inbounds nuw i8, ptr addrspace(3) %176, i64 128
  %195 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) nonnull %194)
  %196 = extractvalue { i32, i32, i32, i32 } %195, 0
  %197 = bitcast i32 %196 to <2 x bfloat>
  %198 = extractelement <2 x bfloat> %197, i64 0
  %199 = extractelement <2 x bfloat> %197, i64 1
  %200 = extractvalue { i32, i32, i32, i32 } %195, 1
  %201 = bitcast i32 %200 to <2 x bfloat>
  %202 = extractelement <2 x bfloat> %201, i64 0
  %203 = extractelement <2 x bfloat> %201, i64 1
  %204 = extractvalue { i32, i32, i32, i32 } %195, 2
  %205 = bitcast i32 %204 to <2 x bfloat>
  %206 = extractelement <2 x bfloat> %205, i64 0
  %207 = extractelement <2 x bfloat> %205, i64 1
  %208 = extractvalue { i32, i32, i32, i32 } %195, 3
  %209 = bitcast i32 %208 to <2 x bfloat>
  %210 = extractelement <2 x bfloat> %209, i64 0
  %211 = extractelement <2 x bfloat> %209, i64 1
  %212 = xor i32 %174, 64
  %213 = zext i32 %212 to i64
  %214 = getelementptr i8, ptr addrspace(3) @global_smem, i64 %213
  %215 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %214)
  %216 = extractvalue { i32, i32, i32, i32 } %215, 0
  %217 = bitcast i32 %216 to <2 x bfloat>
  %218 = extractelement <2 x bfloat> %217, i64 0
  %219 = extractelement <2 x bfloat> %217, i64 1
  %220 = extractvalue { i32, i32, i32, i32 } %215, 1
  %221 = bitcast i32 %220 to <2 x bfloat>
  %222 = extractelement <2 x bfloat> %221, i64 0
  %223 = extractelement <2 x bfloat> %221, i64 1
  %224 = extractvalue { i32, i32, i32, i32 } %215, 2
  %225 = bitcast i32 %224 to <2 x bfloat>
  %226 = extractelement <2 x bfloat> %225, i64 0
  %227 = extractelement <2 x bfloat> %225, i64 1
  %228 = extractvalue { i32, i32, i32, i32 } %215, 3
  %229 = bitcast i32 %228 to <2 x bfloat>
  %230 = extractelement <2 x bfloat> %229, i64 0
  %231 = extractelement <2 x bfloat> %229, i64 1
  %232 = getelementptr inbounds nuw i8, ptr addrspace(3) %214, i64 128
  %233 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) nonnull %232)
  %234 = extractvalue { i32, i32, i32, i32 } %233, 0
  %235 = bitcast i32 %234 to <2 x bfloat>
  %236 = extractelement <2 x bfloat> %235, i64 0
  %237 = extractelement <2 x bfloat> %235, i64 1
  %238 = extractvalue { i32, i32, i32, i32 } %233, 1
  %239 = bitcast i32 %238 to <2 x bfloat>
  %240 = extractelement <2 x bfloat> %239, i64 0
  %241 = extractelement <2 x bfloat> %239, i64 1
  %242 = extractvalue { i32, i32, i32, i32 } %233, 2
  %243 = bitcast i32 %242 to <2 x bfloat>
  %244 = extractelement <2 x bfloat> %243, i64 0
  %245 = extractelement <2 x bfloat> %243, i64 1
  %246 = extractvalue { i32, i32, i32, i32 } %233, 3
  %247 = bitcast i32 %246 to <2 x bfloat>
  %248 = extractelement <2 x bfloat> %247, i64 0
  %249 = extractelement <2 x bfloat> %247, i64 1
  %250 = shl nuw nsw i64 %6, 20
  %251 = trunc i32 %4 to i5
  %252 = zext i5 %251 to i64
  %253 = shl nuw nsw i64 %252, 8
  %254 = add i64 %250, %253
  %255 = add i32 %21, %23
  %256 = zext nneg i32 %255 to i64
  %257 = shl nuw nsw i64 %256, 1
  %258 = add i64 %254, %257
  %scevgep = getelementptr i8, ptr addrspace(1) %2, i64 %258
  %259 = add i64 %250, %.idx319
  %260 = add i32 %14, %11
  %261 = zext nneg i32 %260 to i64
  %262 = shl nuw nsw i64 %261, 1
  %263 = add i64 %259, %262
  %scevgep1027 = getelementptr i8, ptr addrspace(1) %1, i64 %263
  br label %264

264:                                              ; preds = %0, %264
  %lsr.iv = phi i64 [ 0, %0 ], [ %lsr.iv.next, %264 ]
  %.pn65383 = phi bfloat [ %249, %0 ], [ %444, %264 ]
  %.pn67382 = phi bfloat [ %248, %0 ], [ %443, %264 ]
  %.pn69381 = phi bfloat [ %245, %0 ], [ %440, %264 ]
  %.pn71380 = phi bfloat [ %244, %0 ], [ %439, %264 ]
  %.pn73379 = phi bfloat [ %241, %0 ], [ %436, %264 ]
  %.pn75378 = phi bfloat [ %240, %0 ], [ %435, %264 ]
  %.pn77377 = phi bfloat [ %237, %0 ], [ %432, %264 ]
  %.pn79376 = phi bfloat [ %236, %0 ], [ %431, %264 ]
  %.pn81375 = phi bfloat [ %211, %0 ], [ %408, %264 ]
  %.pn83374 = phi bfloat [ %210, %0 ], [ %407, %264 ]
  %.pn85373 = phi bfloat [ %207, %0 ], [ %404, %264 ]
  %.pn87372 = phi bfloat [ %206, %0 ], [ %403, %264 ]
  %.pn89371 = phi bfloat [ %203, %0 ], [ %400, %264 ]
  %.pn91370 = phi bfloat [ %202, %0 ], [ %399, %264 ]
  %.pn93369 = phi bfloat [ %199, %0 ], [ %396, %264 ]
  %.pn95368 = phi bfloat [ %198, %0 ], [ %395, %264 ]
  %.pn97367 = phi bfloat [ %231, %0 ], [ %426, %264 ]
  %.pn99366 = phi bfloat [ %230, %0 ], [ %425, %264 ]
  %.pn101365 = phi bfloat [ %227, %0 ], [ %422, %264 ]
  %.pn103364 = phi bfloat [ %226, %0 ], [ %421, %264 ]
  %.pn105363 = phi bfloat [ %223, %0 ], [ %418, %264 ]
  %.pn107362 = phi bfloat [ %222, %0 ], [ %417, %264 ]
  %.pn109361 = phi bfloat [ %219, %0 ], [ %414, %264 ]
  %.pn111360 = phi bfloat [ %218, %0 ], [ %413, %264 ]
  %.pn113359 = phi bfloat [ %193, %0 ], [ %390, %264 ]
  %.pn115358 = phi bfloat [ %192, %0 ], [ %389, %264 ]
  %.pn117357 = phi bfloat [ %189, %0 ], [ %386, %264 ]
  %.pn119356 = phi bfloat [ %188, %0 ], [ %385, %264 ]
  %.pn121355 = phi bfloat [ %185, %0 ], [ %382, %264 ]
  %.pn123354 = phi bfloat [ %184, %0 ], [ %381, %264 ]
  %.pn125353 = phi bfloat [ %181, %0 ], [ %378, %264 ]
  %.pn127352 = phi bfloat [ %180, %0 ], [ %377, %264 ]
  %.pn1351 = phi bfloat [ %167, %0 ], [ %372, %264 ]
  %.pn3350 = phi bfloat [ %166, %0 ], [ %371, %264 ]
  %.pn5349 = phi bfloat [ %163, %0 ], [ %368, %264 ]
  %.pn7348 = phi bfloat [ %162, %0 ], [ %367, %264 ]
  %.pn9347 = phi bfloat [ %159, %0 ], [ %364, %264 ]
  %.pn11346 = phi bfloat [ %158, %0 ], [ %363, %264 ]
  %.pn13345 = phi bfloat [ %155, %0 ], [ %360, %264 ]
  %.pn15344 = phi bfloat [ %154, %0 ], [ %359, %264 ]
  %.pn17343 = phi bfloat [ %146, %0 ], [ %353, %264 ]
  %.pn19342 = phi bfloat [ %145, %0 ], [ %352, %264 ]
  %.pn21341 = phi bfloat [ %142, %0 ], [ %349, %264 ]
  %.pn23340 = phi bfloat [ %141, %0 ], [ %348, %264 ]
  %.pn25339 = phi bfloat [ %138, %0 ], [ %345, %264 ]
  %.pn27338 = phi bfloat [ %137, %0 ], [ %344, %264 ]
  %.pn29337 = phi bfloat [ %134, %0 ], [ %341, %264 ]
  %.pn31336 = phi bfloat [ %133, %0 ], [ %340, %264 ]
  %.pn33335 = phi bfloat [ %125, %0 ], [ %334, %264 ]
  %.pn35334 = phi bfloat [ %124, %0 ], [ %333, %264 ]
  %.pn37333 = phi bfloat [ %121, %0 ], [ %330, %264 ]
  %.pn39332 = phi bfloat [ %120, %0 ], [ %329, %264 ]
  %.pn41331 = phi bfloat [ %117, %0 ], [ %326, %264 ]
  %.pn43330 = phi bfloat [ %116, %0 ], [ %325, %264 ]
  %.pn45329 = phi bfloat [ %113, %0 ], [ %322, %264 ]
  %.pn47328 = phi bfloat [ %112, %0 ], [ %321, %264 ]
  %.pn49327 = phi bfloat [ %104, %0 ], [ %315, %264 ]
  %.pn51326 = phi bfloat [ %103, %0 ], [ %314, %264 ]
  %.pn53325 = phi bfloat [ %100, %0 ], [ %311, %264 ]
  %.pn55324 = phi bfloat [ %99, %0 ], [ %310, %264 ]
  %.pn57323 = phi bfloat [ %96, %0 ], [ %307, %264 ]
  %.pn59322 = phi bfloat [ %95, %0 ], [ %306, %264 ]
  %.pn61321 = phi bfloat [ %92, %0 ], [ %303, %264 ]
  %.pn63320 = phi bfloat [ %91, %0 ], [ %302, %264 ]
  %265 = phi i32 [ -1, %0 ], [ %272, %264 ]
  %266 = phi i32 [ 2, %0 ], [ %275, %264 ]
  %267 = phi { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } [ zeroinitializer, %0 ], [ %956, %264 ]
  %268 = phi i32 [ 0, %0 ], [ %957, %264 ]
  %269 = icmp samesign ult i32 %268, 5
  %270 = add i32 %265, 1
  %271 = icmp sgt i32 %270, 2
  %272 = select i1 %271, i32 0, i32 %270
  tail call void @llvm.nvvm.cp.async.wait.group(i32 4)
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %273 = add i32 %266, 1
  %274 = icmp sgt i32 %273, 2
  %275 = select i1 %274, i32 0, i32 %273
  %scevgep1030 = getelementptr i8, ptr addrspace(1) %scevgep1027, i64 %lsr.iv
  %scevgep1031 = getelementptr i8, ptr addrspace(1) %scevgep1030, i64 393216
  %scevgep1029 = getelementptr i8, ptr addrspace(1) %scevgep1030, i64 458752
  %276 = shl i32 %275, 10
  %277 = sext i32 %276 to i64
  %278 = getelementptr [2 x i8], ptr addrspace(3) @global_smem, i64 %277
  %279 = getelementptr i8, ptr addrspace(3) %278, i64 %34
  %280 = getelementptr i8, ptr addrspace(3) %279, i64 12288
  %281 = select i1 %269, i32 16, i32 0
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) %280, ptr addrspace(1) %scevgep1031, i32 %281) #7
  %282 = getelementptr i8, ptr addrspace(3) %279, i64 13312
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) nonnull %282, ptr addrspace(1) %scevgep1029, i32 %281) #7
  tail call void @llvm.nvvm.cp.async.commit.group()
  %scevgep1025 = getelementptr i8, ptr addrspace(1) %scevgep, i64 %lsr.iv
  %scevgep1026 = getelementptr i8, ptr addrspace(1) %scevgep1025, i64 393216
  %scevgep1024 = getelementptr i8, ptr addrspace(1) %scevgep1025, i64 425984
  %scevgep1022 = getelementptr i8, ptr addrspace(1) %scevgep1025, i64 458752
  %scevgep1020 = getelementptr i8, ptr addrspace(1) %scevgep1025, i64 491520
  %283 = shl i64 %277, 1
  %284 = getelementptr i8, ptr addrspace(3) %278, i64 %283
  %285 = getelementptr i8, ptr addrspace(3) %284, i64 %45
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) %285, ptr addrspace(1) %scevgep1026, i32 %281) #7
  %286 = getelementptr inbounds nuw i8, ptr addrspace(3) %285, i64 2048
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) nonnull %286, ptr addrspace(1) %scevgep1022, i32 %281) #7
  %287 = getelementptr i8, ptr addrspace(3) %284, i64 %49
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) %287, ptr addrspace(1) %scevgep1024, i32 %281) #7
  %288 = getelementptr inbounds nuw i8, ptr addrspace(3) %287, i64 2048
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) nonnull %288, ptr addrspace(1) %scevgep1020, i32 %281) #7
  tail call void @llvm.nvvm.cp.async.commit.group()
  %289 = add i32 %272, 1
  %290 = icmp sgt i32 %289, 2
  %291 = select i1 %290, i32 0, i32 %289
  %292 = shl i32 %291, 10
  %293 = sext i32 %292 to i64
  %294 = getelementptr [2 x i8], ptr addrspace(3) @global_smem, i64 %293
  %295 = shl i64 %293, 1
  %296 = getelementptr i8, ptr addrspace(3) %294, i64 %295
  tail call void @llvm.nvvm.cp.async.wait.group(i32 4)
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  tail call void @llvm.nvvm.cp.async.wait.group(i32 4)
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %297 = getelementptr i8, ptr addrspace(3) %294, i64 %85
  %298 = getelementptr i8, ptr addrspace(3) %297, i64 12288
  %299 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %298)
  %300 = extractvalue { i32, i32, i32, i32 } %299, 0
  %301 = bitcast i32 %300 to <2 x bfloat>
  %302 = extractelement <2 x bfloat> %301, i64 0
  %303 = extractelement <2 x bfloat> %301, i64 1
  %304 = extractvalue { i32, i32, i32, i32 } %299, 1
  %305 = bitcast i32 %304 to <2 x bfloat>
  %306 = extractelement <2 x bfloat> %305, i64 0
  %307 = extractelement <2 x bfloat> %305, i64 1
  %308 = extractvalue { i32, i32, i32, i32 } %299, 2
  %309 = bitcast i32 %308 to <2 x bfloat>
  %310 = extractelement <2 x bfloat> %309, i64 0
  %311 = extractelement <2 x bfloat> %309, i64 1
  %312 = extractvalue { i32, i32, i32, i32 } %299, 3
  %313 = bitcast i32 %312 to <2 x bfloat>
  %314 = extractelement <2 x bfloat> %313, i64 0
  %315 = extractelement <2 x bfloat> %313, i64 1
  %316 = getelementptr i8, ptr addrspace(3) %294, i64 %106
  %317 = getelementptr i8, ptr addrspace(3) %316, i64 12288
  %318 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %317)
  %319 = extractvalue { i32, i32, i32, i32 } %318, 0
  %320 = bitcast i32 %319 to <2 x bfloat>
  %321 = extractelement <2 x bfloat> %320, i64 0
  %322 = extractelement <2 x bfloat> %320, i64 1
  %323 = extractvalue { i32, i32, i32, i32 } %318, 1
  %324 = bitcast i32 %323 to <2 x bfloat>
  %325 = extractelement <2 x bfloat> %324, i64 0
  %326 = extractelement <2 x bfloat> %324, i64 1
  %327 = extractvalue { i32, i32, i32, i32 } %318, 2
  %328 = bitcast i32 %327 to <2 x bfloat>
  %329 = extractelement <2 x bfloat> %328, i64 0
  %330 = extractelement <2 x bfloat> %328, i64 1
  %331 = extractvalue { i32, i32, i32, i32 } %318, 3
  %332 = bitcast i32 %331 to <2 x bfloat>
  %333 = extractelement <2 x bfloat> %332, i64 0
  %334 = extractelement <2 x bfloat> %332, i64 1
  %335 = getelementptr i8, ptr addrspace(3) %294, i64 %127
  %336 = getelementptr i8, ptr addrspace(3) %335, i64 12288
  %337 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %336)
  %338 = extractvalue { i32, i32, i32, i32 } %337, 0
  %339 = bitcast i32 %338 to <2 x bfloat>
  %340 = extractelement <2 x bfloat> %339, i64 0
  %341 = extractelement <2 x bfloat> %339, i64 1
  %342 = extractvalue { i32, i32, i32, i32 } %337, 1
  %343 = bitcast i32 %342 to <2 x bfloat>
  %344 = extractelement <2 x bfloat> %343, i64 0
  %345 = extractelement <2 x bfloat> %343, i64 1
  %346 = extractvalue { i32, i32, i32, i32 } %337, 2
  %347 = bitcast i32 %346 to <2 x bfloat>
  %348 = extractelement <2 x bfloat> %347, i64 0
  %349 = extractelement <2 x bfloat> %347, i64 1
  %350 = extractvalue { i32, i32, i32, i32 } %337, 3
  %351 = bitcast i32 %350 to <2 x bfloat>
  %352 = extractelement <2 x bfloat> %351, i64 0
  %353 = extractelement <2 x bfloat> %351, i64 1
  %354 = getelementptr i8, ptr addrspace(3) %294, i64 %148
  %355 = getelementptr i8, ptr addrspace(3) %354, i64 12288
  %356 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %355)
  %357 = extractvalue { i32, i32, i32, i32 } %356, 0
  %358 = bitcast i32 %357 to <2 x bfloat>
  %359 = extractelement <2 x bfloat> %358, i64 0
  %360 = extractelement <2 x bfloat> %358, i64 1
  %361 = extractvalue { i32, i32, i32, i32 } %356, 1
  %362 = bitcast i32 %361 to <2 x bfloat>
  %363 = extractelement <2 x bfloat> %362, i64 0
  %364 = extractelement <2 x bfloat> %362, i64 1
  %365 = extractvalue { i32, i32, i32, i32 } %356, 2
  %366 = bitcast i32 %365 to <2 x bfloat>
  %367 = extractelement <2 x bfloat> %366, i64 0
  %368 = extractelement <2 x bfloat> %366, i64 1
  %369 = extractvalue { i32, i32, i32, i32 } %356, 3
  %370 = bitcast i32 %369 to <2 x bfloat>
  %371 = extractelement <2 x bfloat> %370, i64 0
  %372 = extractelement <2 x bfloat> %370, i64 1
  %373 = getelementptr inbounds nuw i8, ptr addrspace(3) %296, i64 %175
  %374 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %373)
  %375 = extractvalue { i32, i32, i32, i32 } %374, 0
  %376 = bitcast i32 %375 to <2 x bfloat>
  %377 = extractelement <2 x bfloat> %376, i64 0
  %378 = extractelement <2 x bfloat> %376, i64 1
  %379 = extractvalue { i32, i32, i32, i32 } %374, 1
  %380 = bitcast i32 %379 to <2 x bfloat>
  %381 = extractelement <2 x bfloat> %380, i64 0
  %382 = extractelement <2 x bfloat> %380, i64 1
  %383 = extractvalue { i32, i32, i32, i32 } %374, 2
  %384 = bitcast i32 %383 to <2 x bfloat>
  %385 = extractelement <2 x bfloat> %384, i64 0
  %386 = extractelement <2 x bfloat> %384, i64 1
  %387 = extractvalue { i32, i32, i32, i32 } %374, 3
  %388 = bitcast i32 %387 to <2 x bfloat>
  %389 = extractelement <2 x bfloat> %388, i64 0
  %390 = extractelement <2 x bfloat> %388, i64 1
  %391 = getelementptr inbounds nuw i8, ptr addrspace(3) %373, i64 128
  %392 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) nonnull %391)
  %393 = extractvalue { i32, i32, i32, i32 } %392, 0
  %394 = bitcast i32 %393 to <2 x bfloat>
  %395 = extractelement <2 x bfloat> %394, i64 0
  %396 = extractelement <2 x bfloat> %394, i64 1
  %397 = extractvalue { i32, i32, i32, i32 } %392, 1
  %398 = bitcast i32 %397 to <2 x bfloat>
  %399 = extractelement <2 x bfloat> %398, i64 0
  %400 = extractelement <2 x bfloat> %398, i64 1
  %401 = extractvalue { i32, i32, i32, i32 } %392, 2
  %402 = bitcast i32 %401 to <2 x bfloat>
  %403 = extractelement <2 x bfloat> %402, i64 0
  %404 = extractelement <2 x bfloat> %402, i64 1
  %405 = extractvalue { i32, i32, i32, i32 } %392, 3
  %406 = bitcast i32 %405 to <2 x bfloat>
  %407 = extractelement <2 x bfloat> %406, i64 0
  %408 = extractelement <2 x bfloat> %406, i64 1
  %409 = getelementptr i8, ptr addrspace(3) %296, i64 %213
  %410 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %409)
  %411 = extractvalue { i32, i32, i32, i32 } %410, 0
  %412 = bitcast i32 %411 to <2 x bfloat>
  %413 = extractelement <2 x bfloat> %412, i64 0
  %414 = extractelement <2 x bfloat> %412, i64 1
  %415 = extractvalue { i32, i32, i32, i32 } %410, 1
  %416 = bitcast i32 %415 to <2 x bfloat>
  %417 = extractelement <2 x bfloat> %416, i64 0
  %418 = extractelement <2 x bfloat> %416, i64 1
  %419 = extractvalue { i32, i32, i32, i32 } %410, 2
  %420 = bitcast i32 %419 to <2 x bfloat>
  %421 = extractelement <2 x bfloat> %420, i64 0
  %422 = extractelement <2 x bfloat> %420, i64 1
  %423 = extractvalue { i32, i32, i32, i32 } %410, 3
  %424 = bitcast i32 %423 to <2 x bfloat>
  %425 = extractelement <2 x bfloat> %424, i64 0
  %426 = extractelement <2 x bfloat> %424, i64 1
  %427 = getelementptr inbounds nuw i8, ptr addrspace(3) %409, i64 128
  %428 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) nonnull %427)
  %429 = extractvalue { i32, i32, i32, i32 } %428, 0
  %430 = bitcast i32 %429 to <2 x bfloat>
  %431 = extractelement <2 x bfloat> %430, i64 0
  %432 = extractelement <2 x bfloat> %430, i64 1
  %433 = extractvalue { i32, i32, i32, i32 } %428, 1
  %434 = bitcast i32 %433 to <2 x bfloat>
  %435 = extractelement <2 x bfloat> %434, i64 0
  %436 = extractelement <2 x bfloat> %434, i64 1
  %437 = extractvalue { i32, i32, i32, i32 } %428, 2
  %438 = bitcast i32 %437 to <2 x bfloat>
  %439 = extractelement <2 x bfloat> %438, i64 0
  %440 = extractelement <2 x bfloat> %438, i64 1
  %441 = extractvalue { i32, i32, i32, i32 } %428, 3
  %442 = bitcast i32 %441 to <2 x bfloat>
  %443 = extractelement <2 x bfloat> %442, i64 0
  %444 = extractelement <2 x bfloat> %442, i64 1
  %445 = insertelement <2 x bfloat> poison, bfloat %.pn63320, i64 0
  %446 = insertelement <2 x bfloat> %445, bfloat %.pn61321, i64 1
  %447 = bitcast <2 x bfloat> %446 to i32
  %448 = insertelement <2 x bfloat> poison, bfloat %.pn59322, i64 0
  %449 = insertelement <2 x bfloat> %448, bfloat %.pn57323, i64 1
  %450 = bitcast <2 x bfloat> %449 to i32
  %451 = insertelement <2 x bfloat> poison, bfloat %.pn55324, i64 0
  %452 = insertelement <2 x bfloat> %451, bfloat %.pn53325, i64 1
  %453 = bitcast <2 x bfloat> %452 to i32
  %454 = insertelement <2 x bfloat> poison, bfloat %.pn51326, i64 0
  %455 = insertelement <2 x bfloat> %454, bfloat %.pn49327, i64 1
  %456 = bitcast <2 x bfloat> %455 to i32
  %457 = insertelement <2 x bfloat> poison, bfloat %.pn47328, i64 0
  %458 = insertelement <2 x bfloat> %457, bfloat %.pn45329, i64 1
  %459 = bitcast <2 x bfloat> %458 to i32
  %460 = insertelement <2 x bfloat> poison, bfloat %.pn43330, i64 0
  %461 = insertelement <2 x bfloat> %460, bfloat %.pn41331, i64 1
  %462 = bitcast <2 x bfloat> %461 to i32
  %463 = insertelement <2 x bfloat> poison, bfloat %.pn39332, i64 0
  %464 = insertelement <2 x bfloat> %463, bfloat %.pn37333, i64 1
  %465 = bitcast <2 x bfloat> %464 to i32
  %466 = insertelement <2 x bfloat> poison, bfloat %.pn35334, i64 0
  %467 = insertelement <2 x bfloat> %466, bfloat %.pn33335, i64 1
  %468 = bitcast <2 x bfloat> %467 to i32
  %469 = insertelement <2 x bfloat> poison, bfloat %.pn31336, i64 0
  %470 = insertelement <2 x bfloat> %469, bfloat %.pn29337, i64 1
  %471 = bitcast <2 x bfloat> %470 to i32
  %472 = insertelement <2 x bfloat> poison, bfloat %.pn27338, i64 0
  %473 = insertelement <2 x bfloat> %472, bfloat %.pn25339, i64 1
  %474 = bitcast <2 x bfloat> %473 to i32
  %475 = insertelement <2 x bfloat> poison, bfloat %.pn23340, i64 0
  %476 = insertelement <2 x bfloat> %475, bfloat %.pn21341, i64 1
  %477 = bitcast <2 x bfloat> %476 to i32
  %478 = insertelement <2 x bfloat> poison, bfloat %.pn19342, i64 0
  %479 = insertelement <2 x bfloat> %478, bfloat %.pn17343, i64 1
  %480 = bitcast <2 x bfloat> %479 to i32
  %481 = insertelement <2 x bfloat> poison, bfloat %.pn15344, i64 0
  %482 = insertelement <2 x bfloat> %481, bfloat %.pn13345, i64 1
  %483 = bitcast <2 x bfloat> %482 to i32
  %484 = insertelement <2 x bfloat> poison, bfloat %.pn11346, i64 0
  %485 = insertelement <2 x bfloat> %484, bfloat %.pn9347, i64 1
  %486 = bitcast <2 x bfloat> %485 to i32
  %487 = insertelement <2 x bfloat> poison, bfloat %.pn7348, i64 0
  %488 = insertelement <2 x bfloat> %487, bfloat %.pn5349, i64 1
  %489 = bitcast <2 x bfloat> %488 to i32
  %490 = insertelement <2 x bfloat> poison, bfloat %.pn3350, i64 0
  %491 = insertelement <2 x bfloat> %490, bfloat %.pn1351, i64 1
  %492 = bitcast <2 x bfloat> %491 to i32
  %493 = insertelement <2 x bfloat> poison, bfloat %.pn127352, i64 0
  %494 = insertelement <2 x bfloat> %493, bfloat %.pn125353, i64 1
  %495 = bitcast <2 x bfloat> %494 to i32
  %496 = insertelement <2 x bfloat> poison, bfloat %.pn123354, i64 0
  %497 = insertelement <2 x bfloat> %496, bfloat %.pn121355, i64 1
  %498 = bitcast <2 x bfloat> %497 to i32
  %499 = insertelement <2 x bfloat> poison, bfloat %.pn119356, i64 0
  %500 = insertelement <2 x bfloat> %499, bfloat %.pn117357, i64 1
  %501 = bitcast <2 x bfloat> %500 to i32
  %502 = insertelement <2 x bfloat> poison, bfloat %.pn115358, i64 0
  %503 = insertelement <2 x bfloat> %502, bfloat %.pn113359, i64 1
  %504 = bitcast <2 x bfloat> %503 to i32
  %505 = insertelement <2 x bfloat> poison, bfloat %.pn111360, i64 0
  %506 = insertelement <2 x bfloat> %505, bfloat %.pn109361, i64 1
  %507 = bitcast <2 x bfloat> %506 to i32
  %508 = insertelement <2 x bfloat> poison, bfloat %.pn107362, i64 0
  %509 = insertelement <2 x bfloat> %508, bfloat %.pn105363, i64 1
  %510 = bitcast <2 x bfloat> %509 to i32
  %511 = insertelement <2 x bfloat> poison, bfloat %.pn103364, i64 0
  %512 = insertelement <2 x bfloat> %511, bfloat %.pn101365, i64 1
  %513 = bitcast <2 x bfloat> %512 to i32
  %514 = insertelement <2 x bfloat> poison, bfloat %.pn99366, i64 0
  %515 = insertelement <2 x bfloat> %514, bfloat %.pn97367, i64 1
  %516 = bitcast <2 x bfloat> %515 to i32
  %517 = insertelement <2 x bfloat> poison, bfloat %.pn95368, i64 0
  %518 = insertelement <2 x bfloat> %517, bfloat %.pn93369, i64 1
  %519 = bitcast <2 x bfloat> %518 to i32
  %520 = insertelement <2 x bfloat> poison, bfloat %.pn91370, i64 0
  %521 = insertelement <2 x bfloat> %520, bfloat %.pn89371, i64 1
  %522 = bitcast <2 x bfloat> %521 to i32
  %523 = insertelement <2 x bfloat> poison, bfloat %.pn87372, i64 0
  %524 = insertelement <2 x bfloat> %523, bfloat %.pn85373, i64 1
  %525 = bitcast <2 x bfloat> %524 to i32
  %526 = insertelement <2 x bfloat> poison, bfloat %.pn83374, i64 0
  %527 = insertelement <2 x bfloat> %526, bfloat %.pn81375, i64 1
  %528 = bitcast <2 x bfloat> %527 to i32
  %529 = insertelement <2 x bfloat> poison, bfloat %.pn79376, i64 0
  %530 = insertelement <2 x bfloat> %529, bfloat %.pn77377, i64 1
  %531 = bitcast <2 x bfloat> %530 to i32
  %532 = insertelement <2 x bfloat> poison, bfloat %.pn75378, i64 0
  %533 = insertelement <2 x bfloat> %532, bfloat %.pn73379, i64 1
  %534 = bitcast <2 x bfloat> %533 to i32
  %535 = insertelement <2 x bfloat> poison, bfloat %.pn71380, i64 0
  %536 = insertelement <2 x bfloat> %535, bfloat %.pn69381, i64 1
  %537 = bitcast <2 x bfloat> %536 to i32
  %538 = insertelement <2 x bfloat> poison, bfloat %.pn67382, i64 0
  %539 = insertelement <2 x bfloat> %538, bfloat %.pn65383, i64 1
  %540 = bitcast <2 x bfloat> %539 to i32
  %541 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 0
  %542 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 1
  %543 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 2
  %544 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 3
  %545 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 4
  %546 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 5
  %547 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 6
  %548 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 7
  %549 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 8
  %550 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 9
  %551 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 10
  %552 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 11
  %553 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 12
  %554 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 13
  %555 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 14
  %556 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 15
  %557 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 16
  %558 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 17
  %559 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 18
  %560 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 19
  %561 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 20
  %562 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 21
  %563 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 22
  %564 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 23
  %565 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 24
  %566 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 25
  %567 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 26
  %568 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 27
  %569 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 28
  %570 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 29
  %571 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 30
  %572 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 31
  %573 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 32
  %574 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 33
  %575 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 34
  %576 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 35
  %577 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 36
  %578 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 37
  %579 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 38
  %580 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 39
  %581 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 40
  %582 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 41
  %583 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 42
  %584 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 43
  %585 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 44
  %586 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 45
  %587 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 46
  %588 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 47
  %589 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 48
  %590 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 49
  %591 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 50
  %592 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 51
  %593 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 52
  %594 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 53
  %595 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 54
  %596 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 55
  %597 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 56
  %598 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 57
  %599 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 58
  %600 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 59
  %601 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 60
  %602 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 61
  %603 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 62
  %604 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 63
  %605 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 64
  %606 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 65
  %607 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 66
  %608 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 67
  %609 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 68
  %610 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 69
  %611 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 70
  %612 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 71
  %613 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 72
  %614 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 73
  %615 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 74
  %616 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 75
  %617 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 76
  %618 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 77
  %619 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 78
  %620 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 79
  %621 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 80
  %622 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 81
  %623 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 82
  %624 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 83
  %625 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 84
  %626 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 85
  %627 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 86
  %628 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 87
  %629 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 88
  %630 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 89
  %631 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 90
  %632 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 91
  %633 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 92
  %634 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 93
  %635 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 94
  %636 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 95
  %637 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 96
  %638 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 97
  %639 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 98
  %640 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 99
  %641 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 100
  %642 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 101
  %643 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 102
  %644 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 103
  %645 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 104
  %646 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 105
  %647 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 106
  %648 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 107
  %649 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 108
  %650 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 109
  %651 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 110
  %652 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 111
  %653 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 112
  %654 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 113
  %655 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 114
  %656 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 115
  %657 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 116
  %658 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 117
  %659 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 118
  %660 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 119
  %661 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 120
  %662 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 121
  %663 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 122
  %664 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 123
  %665 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 124
  %666 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 125
  %667 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 126
  %668 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %267, 127
  %669 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %541, float %542, float %543, float %544, i32 %447, i32 %450, i32 %453, i32 %456, i32 %495, i32 %498) #7
  %670 = extractvalue { float, float, float, float } %669, 0
  %671 = extractvalue { float, float, float, float } %669, 1
  %672 = extractvalue { float, float, float, float } %669, 2
  %673 = extractvalue { float, float, float, float } %669, 3
  %674 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %545, float %546, float %547, float %548, i32 %447, i32 %450, i32 %453, i32 %456, i32 %501, i32 %504) #7
  %675 = extractvalue { float, float, float, float } %674, 0
  %676 = extractvalue { float, float, float, float } %674, 1
  %677 = extractvalue { float, float, float, float } %674, 2
  %678 = extractvalue { float, float, float, float } %674, 3
  %679 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %549, float %550, float %551, float %552, i32 %447, i32 %450, i32 %453, i32 %456, i32 %507, i32 %510) #7
  %680 = extractvalue { float, float, float, float } %679, 0
  %681 = extractvalue { float, float, float, float } %679, 1
  %682 = extractvalue { float, float, float, float } %679, 2
  %683 = extractvalue { float, float, float, float } %679, 3
  %684 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %553, float %554, float %555, float %556, i32 %447, i32 %450, i32 %453, i32 %456, i32 %513, i32 %516) #7
  %685 = extractvalue { float, float, float, float } %684, 0
  %686 = extractvalue { float, float, float, float } %684, 1
  %687 = extractvalue { float, float, float, float } %684, 2
  %688 = extractvalue { float, float, float, float } %684, 3
  %689 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %557, float %558, float %559, float %560, i32 %447, i32 %450, i32 %453, i32 %456, i32 %519, i32 %522) #7
  %690 = extractvalue { float, float, float, float } %689, 0
  %691 = extractvalue { float, float, float, float } %689, 1
  %692 = extractvalue { float, float, float, float } %689, 2
  %693 = extractvalue { float, float, float, float } %689, 3
  %694 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %561, float %562, float %563, float %564, i32 %447, i32 %450, i32 %453, i32 %456, i32 %525, i32 %528) #7
  %695 = extractvalue { float, float, float, float } %694, 0
  %696 = extractvalue { float, float, float, float } %694, 1
  %697 = extractvalue { float, float, float, float } %694, 2
  %698 = extractvalue { float, float, float, float } %694, 3
  %699 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %565, float %566, float %567, float %568, i32 %447, i32 %450, i32 %453, i32 %456, i32 %531, i32 %534) #7
  %700 = extractvalue { float, float, float, float } %699, 0
  %701 = extractvalue { float, float, float, float } %699, 1
  %702 = extractvalue { float, float, float, float } %699, 2
  %703 = extractvalue { float, float, float, float } %699, 3
  %704 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %569, float %570, float %571, float %572, i32 %447, i32 %450, i32 %453, i32 %456, i32 %537, i32 %540) #7
  %705 = extractvalue { float, float, float, float } %704, 0
  %706 = extractvalue { float, float, float, float } %704, 1
  %707 = extractvalue { float, float, float, float } %704, 2
  %708 = extractvalue { float, float, float, float } %704, 3
  %709 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %573, float %574, float %575, float %576, i32 %459, i32 %462, i32 %465, i32 %468, i32 %495, i32 %498) #7
  %710 = extractvalue { float, float, float, float } %709, 0
  %711 = extractvalue { float, float, float, float } %709, 1
  %712 = extractvalue { float, float, float, float } %709, 2
  %713 = extractvalue { float, float, float, float } %709, 3
  %714 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %577, float %578, float %579, float %580, i32 %459, i32 %462, i32 %465, i32 %468, i32 %501, i32 %504) #7
  %715 = extractvalue { float, float, float, float } %714, 0
  %716 = extractvalue { float, float, float, float } %714, 1
  %717 = extractvalue { float, float, float, float } %714, 2
  %718 = extractvalue { float, float, float, float } %714, 3
  %719 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %581, float %582, float %583, float %584, i32 %459, i32 %462, i32 %465, i32 %468, i32 %507, i32 %510) #7
  %720 = extractvalue { float, float, float, float } %719, 0
  %721 = extractvalue { float, float, float, float } %719, 1
  %722 = extractvalue { float, float, float, float } %719, 2
  %723 = extractvalue { float, float, float, float } %719, 3
  %724 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %585, float %586, float %587, float %588, i32 %459, i32 %462, i32 %465, i32 %468, i32 %513, i32 %516) #7
  %725 = extractvalue { float, float, float, float } %724, 0
  %726 = extractvalue { float, float, float, float } %724, 1
  %727 = extractvalue { float, float, float, float } %724, 2
  %728 = extractvalue { float, float, float, float } %724, 3
  %729 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %589, float %590, float %591, float %592, i32 %459, i32 %462, i32 %465, i32 %468, i32 %519, i32 %522) #7
  %730 = extractvalue { float, float, float, float } %729, 0
  %731 = extractvalue { float, float, float, float } %729, 1
  %732 = extractvalue { float, float, float, float } %729, 2
  %733 = extractvalue { float, float, float, float } %729, 3
  %734 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %593, float %594, float %595, float %596, i32 %459, i32 %462, i32 %465, i32 %468, i32 %525, i32 %528) #7
  %735 = extractvalue { float, float, float, float } %734, 0
  %736 = extractvalue { float, float, float, float } %734, 1
  %737 = extractvalue { float, float, float, float } %734, 2
  %738 = extractvalue { float, float, float, float } %734, 3
  %739 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %597, float %598, float %599, float %600, i32 %459, i32 %462, i32 %465, i32 %468, i32 %531, i32 %534) #7
  %740 = extractvalue { float, float, float, float } %739, 0
  %741 = extractvalue { float, float, float, float } %739, 1
  %742 = extractvalue { float, float, float, float } %739, 2
  %743 = extractvalue { float, float, float, float } %739, 3
  %744 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %601, float %602, float %603, float %604, i32 %459, i32 %462, i32 %465, i32 %468, i32 %537, i32 %540) #7
  %745 = extractvalue { float, float, float, float } %744, 0
  %746 = extractvalue { float, float, float, float } %744, 1
  %747 = extractvalue { float, float, float, float } %744, 2
  %748 = extractvalue { float, float, float, float } %744, 3
  %749 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %605, float %606, float %607, float %608, i32 %471, i32 %474, i32 %477, i32 %480, i32 %495, i32 %498) #7
  %750 = extractvalue { float, float, float, float } %749, 0
  %751 = extractvalue { float, float, float, float } %749, 1
  %752 = extractvalue { float, float, float, float } %749, 2
  %753 = extractvalue { float, float, float, float } %749, 3
  %754 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %609, float %610, float %611, float %612, i32 %471, i32 %474, i32 %477, i32 %480, i32 %501, i32 %504) #7
  %755 = extractvalue { float, float, float, float } %754, 0
  %756 = extractvalue { float, float, float, float } %754, 1
  %757 = extractvalue { float, float, float, float } %754, 2
  %758 = extractvalue { float, float, float, float } %754, 3
  %759 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %613, float %614, float %615, float %616, i32 %471, i32 %474, i32 %477, i32 %480, i32 %507, i32 %510) #7
  %760 = extractvalue { float, float, float, float } %759, 0
  %761 = extractvalue { float, float, float, float } %759, 1
  %762 = extractvalue { float, float, float, float } %759, 2
  %763 = extractvalue { float, float, float, float } %759, 3
  %764 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %617, float %618, float %619, float %620, i32 %471, i32 %474, i32 %477, i32 %480, i32 %513, i32 %516) #7
  %765 = extractvalue { float, float, float, float } %764, 0
  %766 = extractvalue { float, float, float, float } %764, 1
  %767 = extractvalue { float, float, float, float } %764, 2
  %768 = extractvalue { float, float, float, float } %764, 3
  %769 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %621, float %622, float %623, float %624, i32 %471, i32 %474, i32 %477, i32 %480, i32 %519, i32 %522) #7
  %770 = extractvalue { float, float, float, float } %769, 0
  %771 = extractvalue { float, float, float, float } %769, 1
  %772 = extractvalue { float, float, float, float } %769, 2
  %773 = extractvalue { float, float, float, float } %769, 3
  %774 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %625, float %626, float %627, float %628, i32 %471, i32 %474, i32 %477, i32 %480, i32 %525, i32 %528) #7
  %775 = extractvalue { float, float, float, float } %774, 0
  %776 = extractvalue { float, float, float, float } %774, 1
  %777 = extractvalue { float, float, float, float } %774, 2
  %778 = extractvalue { float, float, float, float } %774, 3
  %779 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %629, float %630, float %631, float %632, i32 %471, i32 %474, i32 %477, i32 %480, i32 %531, i32 %534) #7
  %780 = extractvalue { float, float, float, float } %779, 0
  %781 = extractvalue { float, float, float, float } %779, 1
  %782 = extractvalue { float, float, float, float } %779, 2
  %783 = extractvalue { float, float, float, float } %779, 3
  %784 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %633, float %634, float %635, float %636, i32 %471, i32 %474, i32 %477, i32 %480, i32 %537, i32 %540) #7
  %785 = extractvalue { float, float, float, float } %784, 0
  %786 = extractvalue { float, float, float, float } %784, 1
  %787 = extractvalue { float, float, float, float } %784, 2
  %788 = extractvalue { float, float, float, float } %784, 3
  %789 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %637, float %638, float %639, float %640, i32 %483, i32 %486, i32 %489, i32 %492, i32 %495, i32 %498) #7
  %790 = extractvalue { float, float, float, float } %789, 0
  %791 = extractvalue { float, float, float, float } %789, 1
  %792 = extractvalue { float, float, float, float } %789, 2
  %793 = extractvalue { float, float, float, float } %789, 3
  %794 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %641, float %642, float %643, float %644, i32 %483, i32 %486, i32 %489, i32 %492, i32 %501, i32 %504) #7
  %795 = extractvalue { float, float, float, float } %794, 0
  %796 = extractvalue { float, float, float, float } %794, 1
  %797 = extractvalue { float, float, float, float } %794, 2
  %798 = extractvalue { float, float, float, float } %794, 3
  %799 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %645, float %646, float %647, float %648, i32 %483, i32 %486, i32 %489, i32 %492, i32 %507, i32 %510) #7
  %800 = extractvalue { float, float, float, float } %799, 0
  %801 = extractvalue { float, float, float, float } %799, 1
  %802 = extractvalue { float, float, float, float } %799, 2
  %803 = extractvalue { float, float, float, float } %799, 3
  %804 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %649, float %650, float %651, float %652, i32 %483, i32 %486, i32 %489, i32 %492, i32 %513, i32 %516) #7
  %805 = extractvalue { float, float, float, float } %804, 0
  %806 = extractvalue { float, float, float, float } %804, 1
  %807 = extractvalue { float, float, float, float } %804, 2
  %808 = extractvalue { float, float, float, float } %804, 3
  %809 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %653, float %654, float %655, float %656, i32 %483, i32 %486, i32 %489, i32 %492, i32 %519, i32 %522) #7
  %810 = extractvalue { float, float, float, float } %809, 0
  %811 = extractvalue { float, float, float, float } %809, 1
  %812 = extractvalue { float, float, float, float } %809, 2
  %813 = extractvalue { float, float, float, float } %809, 3
  %814 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %657, float %658, float %659, float %660, i32 %483, i32 %486, i32 %489, i32 %492, i32 %525, i32 %528) #7
  %815 = extractvalue { float, float, float, float } %814, 0
  %816 = extractvalue { float, float, float, float } %814, 1
  %817 = extractvalue { float, float, float, float } %814, 2
  %818 = extractvalue { float, float, float, float } %814, 3
  %819 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %661, float %662, float %663, float %664, i32 %483, i32 %486, i32 %489, i32 %492, i32 %531, i32 %534) #7
  %820 = extractvalue { float, float, float, float } %819, 0
  %821 = extractvalue { float, float, float, float } %819, 1
  %822 = extractvalue { float, float, float, float } %819, 2
  %823 = extractvalue { float, float, float, float } %819, 3
  %824 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %665, float %666, float %667, float %668, i32 %483, i32 %486, i32 %489, i32 %492, i32 %537, i32 %540) #7
  %825 = extractvalue { float, float, float, float } %824, 0
  %826 = extractvalue { float, float, float, float } %824, 1
  %827 = extractvalue { float, float, float, float } %824, 2
  %828 = extractvalue { float, float, float, float } %824, 3
  %829 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } undef, float %670, 0
  %830 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %829, float %671, 1
  %831 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %830, float %672, 2
  %832 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %831, float %673, 3
  %833 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %832, float %675, 4
  %834 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %833, float %676, 5
  %835 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %834, float %677, 6
  %836 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %835, float %678, 7
  %837 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %836, float %680, 8
  %838 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %837, float %681, 9
  %839 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %838, float %682, 10
  %840 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %839, float %683, 11
  %841 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %840, float %685, 12
  %842 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %841, float %686, 13
  %843 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %842, float %687, 14
  %844 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %843, float %688, 15
  %845 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %844, float %690, 16
  %846 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %845, float %691, 17
  %847 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %846, float %692, 18
  %848 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %847, float %693, 19
  %849 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %848, float %695, 20
  %850 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %849, float %696, 21
  %851 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %850, float %697, 22
  %852 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %851, float %698, 23
  %853 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %852, float %700, 24
  %854 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %853, float %701, 25
  %855 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %854, float %702, 26
  %856 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %855, float %703, 27
  %857 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %856, float %705, 28
  %858 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %857, float %706, 29
  %859 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %858, float %707, 30
  %860 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %859, float %708, 31
  %861 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %860, float %710, 32
  %862 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %861, float %711, 33
  %863 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %862, float %712, 34
  %864 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %863, float %713, 35
  %865 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %864, float %715, 36
  %866 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %865, float %716, 37
  %867 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %866, float %717, 38
  %868 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %867, float %718, 39
  %869 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %868, float %720, 40
  %870 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %869, float %721, 41
  %871 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %870, float %722, 42
  %872 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %871, float %723, 43
  %873 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %872, float %725, 44
  %874 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %873, float %726, 45
  %875 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %874, float %727, 46
  %876 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %875, float %728, 47
  %877 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %876, float %730, 48
  %878 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %877, float %731, 49
  %879 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %878, float %732, 50
  %880 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %879, float %733, 51
  %881 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %880, float %735, 52
  %882 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %881, float %736, 53
  %883 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %882, float %737, 54
  %884 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %883, float %738, 55
  %885 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %884, float %740, 56
  %886 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %885, float %741, 57
  %887 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %886, float %742, 58
  %888 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %887, float %743, 59
  %889 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %888, float %745, 60
  %890 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %889, float %746, 61
  %891 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %890, float %747, 62
  %892 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %891, float %748, 63
  %893 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %892, float %750, 64
  %894 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %893, float %751, 65
  %895 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %894, float %752, 66
  %896 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %895, float %753, 67
  %897 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %896, float %755, 68
  %898 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %897, float %756, 69
  %899 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %898, float %757, 70
  %900 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %899, float %758, 71
  %901 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %900, float %760, 72
  %902 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %901, float %761, 73
  %903 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %902, float %762, 74
  %904 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %903, float %763, 75
  %905 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %904, float %765, 76
  %906 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %905, float %766, 77
  %907 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %906, float %767, 78
  %908 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %907, float %768, 79
  %909 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %908, float %770, 80
  %910 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %909, float %771, 81
  %911 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %910, float %772, 82
  %912 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %911, float %773, 83
  %913 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %912, float %775, 84
  %914 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %913, float %776, 85
  %915 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %914, float %777, 86
  %916 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %915, float %778, 87
  %917 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %916, float %780, 88
  %918 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %917, float %781, 89
  %919 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %918, float %782, 90
  %920 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %919, float %783, 91
  %921 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %920, float %785, 92
  %922 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %921, float %786, 93
  %923 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %922, float %787, 94
  %924 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %923, float %788, 95
  %925 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %924, float %790, 96
  %926 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %925, float %791, 97
  %927 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %926, float %792, 98
  %928 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %927, float %793, 99
  %929 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %928, float %795, 100
  %930 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %929, float %796, 101
  %931 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %930, float %797, 102
  %932 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %931, float %798, 103
  %933 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %932, float %800, 104
  %934 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %933, float %801, 105
  %935 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %934, float %802, 106
  %936 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %935, float %803, 107
  %937 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %936, float %805, 108
  %938 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %937, float %806, 109
  %939 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %938, float %807, 110
  %940 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %939, float %808, 111
  %941 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %940, float %810, 112
  %942 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %941, float %811, 113
  %943 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %942, float %812, 114
  %944 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %943, float %813, 115
  %945 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %944, float %815, 116
  %946 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %945, float %816, 117
  %947 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %946, float %817, 118
  %948 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %947, float %818, 119
  %949 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %948, float %820, 120
  %950 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %949, float %821, 121
  %951 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %950, float %822, 122
  %952 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %951, float %823, 123
  %953 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %952, float %825, 124
  %954 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %953, float %826, 125
  %955 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %954, float %827, 126
  %956 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %955, float %828, 127
  %957 = add nuw nsw i32 %268, 1
  %lsr.iv.next = add nuw nsw i64 %lsr.iv, 131072
  %tmp = trunc i64 %lsr.iv.next to i32
  %exitcond.not = icmp eq i32 %tmp, 1048576
  br i1 %exitcond.not, label %958, label %264

958:                                              ; preds = %264
  tail call void @llvm.nvvm.cp.async.wait.group(i32 0)
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %959 = fptrunc float %670 to bfloat
  %960 = fptrunc float %671 to bfloat
  %961 = fptrunc float %672 to bfloat
  %962 = fptrunc float %673 to bfloat
  %963 = fptrunc float %675 to bfloat
  %964 = fptrunc float %676 to bfloat
  %965 = fptrunc float %677 to bfloat
  %966 = fptrunc float %678 to bfloat
  %967 = fptrunc float %680 to bfloat
  %968 = fptrunc float %681 to bfloat
  %969 = fptrunc float %682 to bfloat
  %970 = fptrunc float %683 to bfloat
  %971 = fptrunc float %685 to bfloat
  %972 = fptrunc float %686 to bfloat
  %973 = fptrunc float %687 to bfloat
  %974 = fptrunc float %688 to bfloat
  %975 = fptrunc float %690 to bfloat
  %976 = fptrunc float %691 to bfloat
  %977 = fptrunc float %692 to bfloat
  %978 = fptrunc float %693 to bfloat
  %979 = fptrunc float %695 to bfloat
  %980 = fptrunc float %696 to bfloat
  %981 = fptrunc float %697 to bfloat
  %982 = fptrunc float %698 to bfloat
  %983 = fptrunc float %700 to bfloat
  %984 = fptrunc float %701 to bfloat
  %985 = fptrunc float %702 to bfloat
  %986 = fptrunc float %703 to bfloat
  %987 = fptrunc float %705 to bfloat
  %988 = fptrunc float %706 to bfloat
  %989 = fptrunc float %707 to bfloat
  %990 = fptrunc float %708 to bfloat
  %991 = fptrunc float %710 to bfloat
  %992 = fptrunc float %711 to bfloat
  %993 = fptrunc float %712 to bfloat
  %994 = fptrunc float %713 to bfloat
  %995 = fptrunc float %715 to bfloat
  %996 = fptrunc float %716 to bfloat
  %997 = fptrunc float %717 to bfloat
  %998 = fptrunc float %718 to bfloat
  %999 = fptrunc float %720 to bfloat
  %1000 = fptrunc float %721 to bfloat
  %1001 = fptrunc float %722 to bfloat
  %1002 = fptrunc float %723 to bfloat
  %1003 = fptrunc float %725 to bfloat
  %1004 = fptrunc float %726 to bfloat
  %1005 = fptrunc float %727 to bfloat
  %1006 = fptrunc float %728 to bfloat
  %1007 = fptrunc float %730 to bfloat
  %1008 = fptrunc float %731 to bfloat
  %1009 = fptrunc float %732 to bfloat
  %1010 = fptrunc float %733 to bfloat
  %1011 = fptrunc float %735 to bfloat
  %1012 = fptrunc float %736 to bfloat
  %1013 = fptrunc float %737 to bfloat
  %1014 = fptrunc float %738 to bfloat
  %1015 = fptrunc float %740 to bfloat
  %1016 = fptrunc float %741 to bfloat
  %1017 = fptrunc float %742 to bfloat
  %1018 = fptrunc float %743 to bfloat
  %1019 = fptrunc float %745 to bfloat
  %1020 = fptrunc float %746 to bfloat
  %1021 = fptrunc float %747 to bfloat
  %1022 = fptrunc float %748 to bfloat
  %1023 = fptrunc float %750 to bfloat
  %1024 = fptrunc float %751 to bfloat
  %1025 = fptrunc float %752 to bfloat
  %1026 = fptrunc float %753 to bfloat
  %1027 = fptrunc float %755 to bfloat
  %1028 = fptrunc float %756 to bfloat
  %1029 = fptrunc float %757 to bfloat
  %1030 = fptrunc float %758 to bfloat
  %1031 = fptrunc float %760 to bfloat
  %1032 = fptrunc float %761 to bfloat
  %1033 = fptrunc float %762 to bfloat
  %1034 = fptrunc float %763 to bfloat
  %1035 = fptrunc float %765 to bfloat
  %1036 = fptrunc float %766 to bfloat
  %1037 = fptrunc float %767 to bfloat
  %1038 = fptrunc float %768 to bfloat
  %1039 = fptrunc float %770 to bfloat
  %1040 = fptrunc float %771 to bfloat
  %1041 = fptrunc float %772 to bfloat
  %1042 = fptrunc float %773 to bfloat
  %1043 = fptrunc float %775 to bfloat
  %1044 = fptrunc float %776 to bfloat
  %1045 = fptrunc float %777 to bfloat
  %1046 = fptrunc float %778 to bfloat
  %1047 = fptrunc float %780 to bfloat
  %1048 = fptrunc float %781 to bfloat
  %1049 = fptrunc float %782 to bfloat
  %1050 = fptrunc float %783 to bfloat
  %1051 = fptrunc float %785 to bfloat
  %1052 = fptrunc float %786 to bfloat
  %1053 = fptrunc float %787 to bfloat
  %1054 = fptrunc float %788 to bfloat
  %1055 = fptrunc float %790 to bfloat
  %1056 = fptrunc float %791 to bfloat
  %1057 = fptrunc float %792 to bfloat
  %1058 = fptrunc float %793 to bfloat
  %1059 = fptrunc float %795 to bfloat
  %1060 = fptrunc float %796 to bfloat
  %1061 = fptrunc float %797 to bfloat
  %1062 = fptrunc float %798 to bfloat
  %1063 = fptrunc float %800 to bfloat
  %1064 = fptrunc float %801 to bfloat
  %1065 = fptrunc float %802 to bfloat
  %1066 = fptrunc float %803 to bfloat
  %1067 = fptrunc float %805 to bfloat
  %1068 = fptrunc float %806 to bfloat
  %1069 = fptrunc float %807 to bfloat
  %1070 = fptrunc float %808 to bfloat
  %1071 = fptrunc float %810 to bfloat
  %1072 = fptrunc float %811 to bfloat
  %1073 = fptrunc float %812 to bfloat
  %1074 = fptrunc float %813 to bfloat
  %1075 = fptrunc float %815 to bfloat
  %1076 = fptrunc float %816 to bfloat
  %1077 = fptrunc float %817 to bfloat
  %1078 = fptrunc float %818 to bfloat
  %1079 = fptrunc float %820 to bfloat
  %1080 = fptrunc float %821 to bfloat
  %1081 = fptrunc float %822 to bfloat
  %1082 = fptrunc float %823 to bfloat
  %1083 = fptrunc float %825 to bfloat
  %1084 = fptrunc float %826 to bfloat
  %1085 = fptrunc float %827 to bfloat
  %1086 = fptrunc float %828 to bfloat
  %1087 = and i32 %9, 3
  %1088 = shl nuw nsw i32 %1087, 11
  %1089 = shl nuw nsw i32 %1087, 5
  %1090 = and i32 %30, 384
  %1091 = and i32 %9, 4
  %1092 = icmp eq i32 %1091, 0
  %1093 = select i1 %1092, i32 0, i32 1040
  %1094 = or disjoint i32 %1089, %1090
  %1095 = xor i32 %1093, %171
  %1096 = or disjoint i32 %1095, %1094
  %1097 = or disjoint i32 %1096, %1088
  %1098 = zext nneg i32 %1097 to i64
  %1099 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %1098
  %1100 = insertelement <2 x bfloat> poison, bfloat %959, i64 0
  %1101 = insertelement <2 x bfloat> %1100, bfloat %960, i64 1
  %1102 = bitcast <2 x bfloat> %1101 to i32
  %1103 = insertelement <2 x bfloat> poison, bfloat %961, i64 0
  %1104 = insertelement <2 x bfloat> %1103, bfloat %962, i64 1
  %1105 = bitcast <2 x bfloat> %1104 to i32
  %1106 = insertelement <2 x bfloat> poison, bfloat %991, i64 0
  %1107 = insertelement <2 x bfloat> %1106, bfloat %992, i64 1
  %1108 = bitcast <2 x bfloat> %1107 to i32
  %1109 = insertelement <2 x bfloat> poison, bfloat %993, i64 0
  %1110 = insertelement <2 x bfloat> %1109, bfloat %994, i64 1
  %1111 = bitcast <2 x bfloat> %1110 to i32
  %1112 = insertelement <4 x i32> poison, i32 %1102, i64 0
  %1113 = insertelement <4 x i32> %1112, i32 %1105, i64 1
  %1114 = insertelement <4 x i32> %1113, i32 %1108, i64 2
  %1115 = insertelement <4 x i32> %1114, i32 %1111, i64 3
  store <4 x i32> %1115, ptr addrspace(3) %1099, align 16
  %1116 = getelementptr inbounds nuw i8, ptr addrspace(3) %1099, i64 512
  %1117 = insertelement <2 x bfloat> poison, bfloat %975, i64 0
  %1118 = insertelement <2 x bfloat> %1117, bfloat %976, i64 1
  %1119 = bitcast <2 x bfloat> %1118 to i32
  %1120 = insertelement <2 x bfloat> poison, bfloat %977, i64 0
  %1121 = insertelement <2 x bfloat> %1120, bfloat %978, i64 1
  %1122 = bitcast <2 x bfloat> %1121 to i32
  %1123 = insertelement <2 x bfloat> poison, bfloat %1007, i64 0
  %1124 = insertelement <2 x bfloat> %1123, bfloat %1008, i64 1
  %1125 = bitcast <2 x bfloat> %1124 to i32
  %1126 = insertelement <2 x bfloat> poison, bfloat %1009, i64 0
  %1127 = insertelement <2 x bfloat> %1126, bfloat %1010, i64 1
  %1128 = bitcast <2 x bfloat> %1127 to i32
  %1129 = insertelement <4 x i32> poison, i32 %1119, i64 0
  %1130 = insertelement <4 x i32> %1129, i32 %1122, i64 1
  %1131 = insertelement <4 x i32> %1130, i32 %1125, i64 2
  %1132 = insertelement <4 x i32> %1131, i32 %1128, i64 3
  store <4 x i32> %1132, ptr addrspace(3) %1116, align 16
  %1133 = xor i32 %1097, 32
  %1134 = zext nneg i32 %1133 to i64
  %1135 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %1134
  %1136 = insertelement <2 x bfloat> poison, bfloat %963, i64 0
  %1137 = insertelement <2 x bfloat> %1136, bfloat %964, i64 1
  %1138 = bitcast <2 x bfloat> %1137 to i32
  %1139 = insertelement <2 x bfloat> poison, bfloat %965, i64 0
  %1140 = insertelement <2 x bfloat> %1139, bfloat %966, i64 1
  %1141 = bitcast <2 x bfloat> %1140 to i32
  %1142 = insertelement <2 x bfloat> poison, bfloat %995, i64 0
  %1143 = insertelement <2 x bfloat> %1142, bfloat %996, i64 1
  %1144 = bitcast <2 x bfloat> %1143 to i32
  %1145 = insertelement <2 x bfloat> poison, bfloat %997, i64 0
  %1146 = insertelement <2 x bfloat> %1145, bfloat %998, i64 1
  %1147 = bitcast <2 x bfloat> %1146 to i32
  %1148 = insertelement <4 x i32> poison, i32 %1138, i64 0
  %1149 = insertelement <4 x i32> %1148, i32 %1141, i64 1
  %1150 = insertelement <4 x i32> %1149, i32 %1144, i64 2
  %1151 = insertelement <4 x i32> %1150, i32 %1147, i64 3
  store <4 x i32> %1151, ptr addrspace(3) %1135, align 16
  %1152 = getelementptr inbounds nuw i8, ptr addrspace(3) %1135, i64 512
  %1153 = insertelement <2 x bfloat> poison, bfloat %979, i64 0
  %1154 = insertelement <2 x bfloat> %1153, bfloat %980, i64 1
  %1155 = bitcast <2 x bfloat> %1154 to i32
  %1156 = insertelement <2 x bfloat> poison, bfloat %981, i64 0
  %1157 = insertelement <2 x bfloat> %1156, bfloat %982, i64 1
  %1158 = bitcast <2 x bfloat> %1157 to i32
  %1159 = insertelement <2 x bfloat> poison, bfloat %1011, i64 0
  %1160 = insertelement <2 x bfloat> %1159, bfloat %1012, i64 1
  %1161 = bitcast <2 x bfloat> %1160 to i32
  %1162 = insertelement <2 x bfloat> poison, bfloat %1013, i64 0
  %1163 = insertelement <2 x bfloat> %1162, bfloat %1014, i64 1
  %1164 = bitcast <2 x bfloat> %1163 to i32
  %1165 = insertelement <4 x i32> poison, i32 %1155, i64 0
  %1166 = insertelement <4 x i32> %1165, i32 %1158, i64 1
  %1167 = insertelement <4 x i32> %1166, i32 %1161, i64 2
  %1168 = insertelement <4 x i32> %1167, i32 %1164, i64 3
  store <4 x i32> %1168, ptr addrspace(3) %1152, align 16
  %1169 = xor i32 %1097, 64
  %1170 = zext nneg i32 %1169 to i64
  %1171 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %1170
  %1172 = insertelement <2 x bfloat> poison, bfloat %967, i64 0
  %1173 = insertelement <2 x bfloat> %1172, bfloat %968, i64 1
  %1174 = bitcast <2 x bfloat> %1173 to i32
  %1175 = insertelement <2 x bfloat> poison, bfloat %969, i64 0
  %1176 = insertelement <2 x bfloat> %1175, bfloat %970, i64 1
  %1177 = bitcast <2 x bfloat> %1176 to i32
  %1178 = insertelement <2 x bfloat> poison, bfloat %999, i64 0
  %1179 = insertelement <2 x bfloat> %1178, bfloat %1000, i64 1
  %1180 = bitcast <2 x bfloat> %1179 to i32
  %1181 = insertelement <2 x bfloat> poison, bfloat %1001, i64 0
  %1182 = insertelement <2 x bfloat> %1181, bfloat %1002, i64 1
  %1183 = bitcast <2 x bfloat> %1182 to i32
  %1184 = insertelement <4 x i32> poison, i32 %1174, i64 0
  %1185 = insertelement <4 x i32> %1184, i32 %1177, i64 1
  %1186 = insertelement <4 x i32> %1185, i32 %1180, i64 2
  %1187 = insertelement <4 x i32> %1186, i32 %1183, i64 3
  store <4 x i32> %1187, ptr addrspace(3) %1171, align 16
  %1188 = getelementptr inbounds nuw i8, ptr addrspace(3) %1171, i64 512
  %1189 = insertelement <2 x bfloat> poison, bfloat %983, i64 0
  %1190 = insertelement <2 x bfloat> %1189, bfloat %984, i64 1
  %1191 = bitcast <2 x bfloat> %1190 to i32
  %1192 = insertelement <2 x bfloat> poison, bfloat %985, i64 0
  %1193 = insertelement <2 x bfloat> %1192, bfloat %986, i64 1
  %1194 = bitcast <2 x bfloat> %1193 to i32
  %1195 = insertelement <2 x bfloat> poison, bfloat %1015, i64 0
  %1196 = insertelement <2 x bfloat> %1195, bfloat %1016, i64 1
  %1197 = bitcast <2 x bfloat> %1196 to i32
  %1198 = insertelement <2 x bfloat> poison, bfloat %1017, i64 0
  %1199 = insertelement <2 x bfloat> %1198, bfloat %1018, i64 1
  %1200 = bitcast <2 x bfloat> %1199 to i32
  %1201 = insertelement <4 x i32> poison, i32 %1191, i64 0
  %1202 = insertelement <4 x i32> %1201, i32 %1194, i64 1
  %1203 = insertelement <4 x i32> %1202, i32 %1197, i64 2
  %1204 = insertelement <4 x i32> %1203, i32 %1200, i64 3
  store <4 x i32> %1204, ptr addrspace(3) %1188, align 16
  %1205 = xor i32 %1097, 96
  %1206 = zext nneg i32 %1205 to i64
  %1207 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %1206
  %1208 = insertelement <2 x bfloat> poison, bfloat %971, i64 0
  %1209 = insertelement <2 x bfloat> %1208, bfloat %972, i64 1
  %1210 = bitcast <2 x bfloat> %1209 to i32
  %1211 = insertelement <2 x bfloat> poison, bfloat %973, i64 0
  %1212 = insertelement <2 x bfloat> %1211, bfloat %974, i64 1
  %1213 = bitcast <2 x bfloat> %1212 to i32
  %1214 = insertelement <2 x bfloat> poison, bfloat %1003, i64 0
  %1215 = insertelement <2 x bfloat> %1214, bfloat %1004, i64 1
  %1216 = bitcast <2 x bfloat> %1215 to i32
  %1217 = insertelement <2 x bfloat> poison, bfloat %1005, i64 0
  %1218 = insertelement <2 x bfloat> %1217, bfloat %1006, i64 1
  %1219 = bitcast <2 x bfloat> %1218 to i32
  %1220 = insertelement <4 x i32> poison, i32 %1210, i64 0
  %1221 = insertelement <4 x i32> %1220, i32 %1213, i64 1
  %1222 = insertelement <4 x i32> %1221, i32 %1216, i64 2
  %1223 = insertelement <4 x i32> %1222, i32 %1219, i64 3
  store <4 x i32> %1223, ptr addrspace(3) %1207, align 16
  %1224 = getelementptr inbounds nuw i8, ptr addrspace(3) %1207, i64 512
  %1225 = insertelement <2 x bfloat> poison, bfloat %987, i64 0
  %1226 = insertelement <2 x bfloat> %1225, bfloat %988, i64 1
  %1227 = bitcast <2 x bfloat> %1226 to i32
  %1228 = insertelement <2 x bfloat> poison, bfloat %989, i64 0
  %1229 = insertelement <2 x bfloat> %1228, bfloat %990, i64 1
  %1230 = bitcast <2 x bfloat> %1229 to i32
  %1231 = insertelement <2 x bfloat> poison, bfloat %1019, i64 0
  %1232 = insertelement <2 x bfloat> %1231, bfloat %1020, i64 1
  %1233 = bitcast <2 x bfloat> %1232 to i32
  %1234 = insertelement <2 x bfloat> poison, bfloat %1021, i64 0
  %1235 = insertelement <2 x bfloat> %1234, bfloat %1022, i64 1
  %1236 = bitcast <2 x bfloat> %1235 to i32
  %1237 = insertelement <4 x i32> poison, i32 %1227, i64 0
  %1238 = insertelement <4 x i32> %1237, i32 %1230, i64 1
  %1239 = insertelement <4 x i32> %1238, i32 %1233, i64 2
  %1240 = insertelement <4 x i32> %1239, i32 %1236, i64 3
  store <4 x i32> %1240, ptr addrspace(3) %1224, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %1241 = shl nuw nsw i32 %78, 6
  %1242 = icmp eq i32 %80, 0
  %1243 = select i1 %1242, i32 0, i32 1040
  %1244 = shl nuw nsw i32 %170, 2
  %1245 = or disjoint i32 %1241, %1244
  %1246 = xor i32 %1243, %77
  %1247 = or disjoint i32 %1246, %1245
  %1248 = zext nneg i32 %1247 to i64
  %1249 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %1248
  %1250 = getelementptr inbounds nuw i8, ptr addrspace(3) %1249, i64 256
  %1251 = xor i32 %1247, 2080
  %1252 = zext nneg i32 %1251 to i64
  %1253 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %1252
  %1254 = getelementptr inbounds nuw i8, ptr addrspace(3) %1253, i64 256
  %1255 = xor i32 %1247, 4160
  %1256 = zext nneg i32 %1255 to i64
  %1257 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %1256
  %1258 = getelementptr inbounds nuw i8, ptr addrspace(3) %1257, i64 256
  %1259 = xor i32 %1247, 6240
  %1260 = zext nneg i32 %1259 to i64
  %1261 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %1260
  %1262 = getelementptr inbounds nuw i8, ptr addrspace(3) %1261, i64 256
  %1263 = insertelement <2 x bfloat> poison, bfloat %1023, i64 0
  %1264 = insertelement <2 x bfloat> %1263, bfloat %1024, i64 1
  %1265 = bitcast <2 x bfloat> %1264 to i32
  %1266 = insertelement <2 x bfloat> poison, bfloat %1025, i64 0
  %1267 = insertelement <2 x bfloat> %1266, bfloat %1026, i64 1
  %1268 = bitcast <2 x bfloat> %1267 to i32
  %1269 = insertelement <2 x bfloat> poison, bfloat %1055, i64 0
  %1270 = insertelement <2 x bfloat> %1269, bfloat %1056, i64 1
  %1271 = bitcast <2 x bfloat> %1270 to i32
  %1272 = insertelement <2 x bfloat> poison, bfloat %1057, i64 0
  %1273 = insertelement <2 x bfloat> %1272, bfloat %1058, i64 1
  %1274 = bitcast <2 x bfloat> %1273 to i32
  %1275 = insertelement <4 x i32> poison, i32 %1265, i64 0
  %1276 = insertelement <4 x i32> %1275, i32 %1268, i64 1
  %1277 = insertelement <4 x i32> %1276, i32 %1271, i64 2
  %1278 = insertelement <4 x i32> %1277, i32 %1274, i64 3
  %1279 = insertelement <2 x bfloat> poison, bfloat %1039, i64 0
  %1280 = insertelement <2 x bfloat> %1279, bfloat %1040, i64 1
  %1281 = bitcast <2 x bfloat> %1280 to i32
  %1282 = insertelement <2 x bfloat> poison, bfloat %1041, i64 0
  %1283 = insertelement <2 x bfloat> %1282, bfloat %1042, i64 1
  %1284 = bitcast <2 x bfloat> %1283 to i32
  %1285 = insertelement <2 x bfloat> poison, bfloat %1071, i64 0
  %1286 = insertelement <2 x bfloat> %1285, bfloat %1072, i64 1
  %1287 = bitcast <2 x bfloat> %1286 to i32
  %1288 = insertelement <2 x bfloat> poison, bfloat %1073, i64 0
  %1289 = insertelement <2 x bfloat> %1288, bfloat %1074, i64 1
  %1290 = bitcast <2 x bfloat> %1289 to i32
  %1291 = insertelement <4 x i32> poison, i32 %1281, i64 0
  %1292 = insertelement <4 x i32> %1291, i32 %1284, i64 1
  %1293 = insertelement <4 x i32> %1292, i32 %1287, i64 2
  %1294 = insertelement <4 x i32> %1293, i32 %1290, i64 3
  %1295 = insertelement <2 x bfloat> poison, bfloat %1027, i64 0
  %1296 = insertelement <2 x bfloat> %1295, bfloat %1028, i64 1
  %1297 = bitcast <2 x bfloat> %1296 to i32
  %1298 = insertelement <2 x bfloat> poison, bfloat %1029, i64 0
  %1299 = insertelement <2 x bfloat> %1298, bfloat %1030, i64 1
  %1300 = bitcast <2 x bfloat> %1299 to i32
  %1301 = insertelement <2 x bfloat> poison, bfloat %1059, i64 0
  %1302 = insertelement <2 x bfloat> %1301, bfloat %1060, i64 1
  %1303 = bitcast <2 x bfloat> %1302 to i32
  %1304 = insertelement <2 x bfloat> poison, bfloat %1061, i64 0
  %1305 = insertelement <2 x bfloat> %1304, bfloat %1062, i64 1
  %1306 = bitcast <2 x bfloat> %1305 to i32
  %1307 = insertelement <4 x i32> poison, i32 %1297, i64 0
  %1308 = insertelement <4 x i32> %1307, i32 %1300, i64 1
  %1309 = insertelement <4 x i32> %1308, i32 %1303, i64 2
  %1310 = insertelement <4 x i32> %1309, i32 %1306, i64 3
  %1311 = insertelement <2 x bfloat> poison, bfloat %1043, i64 0
  %1312 = insertelement <2 x bfloat> %1311, bfloat %1044, i64 1
  %1313 = bitcast <2 x bfloat> %1312 to i32
  %1314 = insertelement <2 x bfloat> poison, bfloat %1045, i64 0
  %1315 = insertelement <2 x bfloat> %1314, bfloat %1046, i64 1
  %1316 = bitcast <2 x bfloat> %1315 to i32
  %1317 = insertelement <2 x bfloat> poison, bfloat %1075, i64 0
  %1318 = insertelement <2 x bfloat> %1317, bfloat %1076, i64 1
  %1319 = bitcast <2 x bfloat> %1318 to i32
  %1320 = insertelement <2 x bfloat> poison, bfloat %1077, i64 0
  %1321 = insertelement <2 x bfloat> %1320, bfloat %1078, i64 1
  %1322 = bitcast <2 x bfloat> %1321 to i32
  %1323 = insertelement <4 x i32> poison, i32 %1313, i64 0
  %1324 = insertelement <4 x i32> %1323, i32 %1316, i64 1
  %1325 = insertelement <4 x i32> %1324, i32 %1319, i64 2
  %1326 = insertelement <4 x i32> %1325, i32 %1322, i64 3
  %1327 = insertelement <2 x bfloat> poison, bfloat %1031, i64 0
  %1328 = insertelement <2 x bfloat> %1327, bfloat %1032, i64 1
  %1329 = bitcast <2 x bfloat> %1328 to i32
  %1330 = insertelement <2 x bfloat> poison, bfloat %1033, i64 0
  %1331 = insertelement <2 x bfloat> %1330, bfloat %1034, i64 1
  %1332 = bitcast <2 x bfloat> %1331 to i32
  %1333 = insertelement <2 x bfloat> poison, bfloat %1063, i64 0
  %1334 = insertelement <2 x bfloat> %1333, bfloat %1064, i64 1
  %1335 = bitcast <2 x bfloat> %1334 to i32
  %1336 = insertelement <2 x bfloat> poison, bfloat %1065, i64 0
  %1337 = insertelement <2 x bfloat> %1336, bfloat %1066, i64 1
  %1338 = bitcast <2 x bfloat> %1337 to i32
  %1339 = insertelement <4 x i32> poison, i32 %1329, i64 0
  %1340 = insertelement <4 x i32> %1339, i32 %1332, i64 1
  %1341 = insertelement <4 x i32> %1340, i32 %1335, i64 2
  %1342 = insertelement <4 x i32> %1341, i32 %1338, i64 3
  %1343 = insertelement <2 x bfloat> poison, bfloat %1047, i64 0
  %1344 = insertelement <2 x bfloat> %1343, bfloat %1048, i64 1
  %1345 = bitcast <2 x bfloat> %1344 to i32
  %1346 = insertelement <2 x bfloat> poison, bfloat %1049, i64 0
  %1347 = insertelement <2 x bfloat> %1346, bfloat %1050, i64 1
  %1348 = bitcast <2 x bfloat> %1347 to i32
  %1349 = insertelement <2 x bfloat> poison, bfloat %1079, i64 0
  %1350 = insertelement <2 x bfloat> %1349, bfloat %1080, i64 1
  %1351 = bitcast <2 x bfloat> %1350 to i32
  %1352 = insertelement <2 x bfloat> poison, bfloat %1081, i64 0
  %1353 = insertelement <2 x bfloat> %1352, bfloat %1082, i64 1
  %1354 = bitcast <2 x bfloat> %1353 to i32
  %1355 = insertelement <4 x i32> poison, i32 %1345, i64 0
  %1356 = insertelement <4 x i32> %1355, i32 %1348, i64 1
  %1357 = insertelement <4 x i32> %1356, i32 %1351, i64 2
  %1358 = insertelement <4 x i32> %1357, i32 %1354, i64 3
  %1359 = insertelement <2 x bfloat> poison, bfloat %1035, i64 0
  %1360 = insertelement <2 x bfloat> %1359, bfloat %1036, i64 1
  %1361 = bitcast <2 x bfloat> %1360 to i32
  %1362 = insertelement <2 x bfloat> poison, bfloat %1037, i64 0
  %1363 = insertelement <2 x bfloat> %1362, bfloat %1038, i64 1
  %1364 = bitcast <2 x bfloat> %1363 to i32
  %1365 = insertelement <2 x bfloat> poison, bfloat %1067, i64 0
  %1366 = insertelement <2 x bfloat> %1365, bfloat %1068, i64 1
  %1367 = bitcast <2 x bfloat> %1366 to i32
  %1368 = insertelement <2 x bfloat> poison, bfloat %1069, i64 0
  %1369 = insertelement <2 x bfloat> %1368, bfloat %1070, i64 1
  %1370 = bitcast <2 x bfloat> %1369 to i32
  %1371 = insertelement <4 x i32> poison, i32 %1361, i64 0
  %1372 = insertelement <4 x i32> %1371, i32 %1364, i64 1
  %1373 = insertelement <4 x i32> %1372, i32 %1367, i64 2
  %1374 = insertelement <4 x i32> %1373, i32 %1370, i64 3
  %1375 = insertelement <2 x bfloat> poison, bfloat %1051, i64 0
  %1376 = insertelement <2 x bfloat> %1375, bfloat %1052, i64 1
  %1377 = bitcast <2 x bfloat> %1376 to i32
  %1378 = insertelement <2 x bfloat> poison, bfloat %1053, i64 0
  %1379 = insertelement <2 x bfloat> %1378, bfloat %1054, i64 1
  %1380 = bitcast <2 x bfloat> %1379 to i32
  %1381 = insertelement <2 x bfloat> poison, bfloat %1083, i64 0
  %1382 = insertelement <2 x bfloat> %1381, bfloat %1084, i64 1
  %1383 = bitcast <2 x bfloat> %1382 to i32
  %1384 = insertelement <2 x bfloat> poison, bfloat %1085, i64 0
  %1385 = insertelement <2 x bfloat> %1384, bfloat %1086, i64 1
  %1386 = bitcast <2 x bfloat> %1385 to i32
  %1387 = insertelement <4 x i32> poison, i32 %1377, i64 0
  %1388 = insertelement <4 x i32> %1387, i32 %1380, i64 1
  %1389 = insertelement <4 x i32> %1388, i32 %1383, i64 2
  %1390 = insertelement <4 x i32> %1389, i32 %1386, i64 3
  %1391 = load <8 x bfloat>, ptr addrspace(3) %1249, align 16
  %1392 = shufflevector <8 x bfloat> %1391, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1393 = shufflevector <8 x bfloat> %1391, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1394 = shufflevector <8 x bfloat> %1391, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1395 = shufflevector <8 x bfloat> %1391, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1396 = fmul <2 x bfloat> %1392, splat (bfloat 0xR3DB5)
  %1397 = load <8 x bfloat>, ptr addrspace(3) %1253, align 16
  %1398 = shufflevector <8 x bfloat> %1397, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1399 = shufflevector <8 x bfloat> %1397, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1400 = shufflevector <8 x bfloat> %1397, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1401 = shufflevector <8 x bfloat> %1397, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1402 = fmul <2 x bfloat> %1398, splat (bfloat 0xR3DB5)
  %1403 = load <8 x bfloat>, ptr addrspace(3) %1257, align 16
  %1404 = shufflevector <8 x bfloat> %1403, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1405 = shufflevector <8 x bfloat> %1403, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1406 = shufflevector <8 x bfloat> %1403, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1407 = shufflevector <8 x bfloat> %1403, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1408 = fmul <2 x bfloat> %1404, splat (bfloat 0xR3DB5)
  %1409 = load <8 x bfloat>, ptr addrspace(3) %1261, align 16
  %1410 = shufflevector <8 x bfloat> %1409, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1411 = shufflevector <8 x bfloat> %1409, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1412 = shufflevector <8 x bfloat> %1409, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1413 = shufflevector <8 x bfloat> %1409, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1414 = fmul <2 x bfloat> %1410, splat (bfloat 0xR3DB5)
  %1415 = load <8 x bfloat>, ptr addrspace(3) %1250, align 16
  %1416 = shufflevector <8 x bfloat> %1415, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1417 = shufflevector <8 x bfloat> %1415, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1418 = shufflevector <8 x bfloat> %1415, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1419 = shufflevector <8 x bfloat> %1415, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1420 = fmul <2 x bfloat> %1416, splat (bfloat 0xR3DB5)
  %1421 = load <8 x bfloat>, ptr addrspace(3) %1254, align 16
  %1422 = shufflevector <8 x bfloat> %1421, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1423 = shufflevector <8 x bfloat> %1421, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1424 = shufflevector <8 x bfloat> %1421, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1425 = shufflevector <8 x bfloat> %1421, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1426 = fmul <2 x bfloat> %1422, splat (bfloat 0xR3DB5)
  %1427 = load <8 x bfloat>, ptr addrspace(3) %1258, align 16
  %1428 = shufflevector <8 x bfloat> %1427, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1429 = shufflevector <8 x bfloat> %1427, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1430 = shufflevector <8 x bfloat> %1427, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1431 = shufflevector <8 x bfloat> %1427, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1432 = fmul <2 x bfloat> %1428, splat (bfloat 0xR3DB5)
  %1433 = load <8 x bfloat>, ptr addrspace(3) %1262, align 16
  %1434 = shufflevector <8 x bfloat> %1433, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1435 = shufflevector <8 x bfloat> %1433, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1436 = shufflevector <8 x bfloat> %1433, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1437 = shufflevector <8 x bfloat> %1433, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1438 = fmul <2 x bfloat> %1434, splat (bfloat 0xR3DB5)
  %1439 = fmul <2 x bfloat> %1393, splat (bfloat 0xR3DB5)
  %1440 = fmul <2 x bfloat> %1399, splat (bfloat 0xR3DB5)
  %1441 = fmul <2 x bfloat> %1405, splat (bfloat 0xR3DB5)
  %1442 = fmul <2 x bfloat> %1411, splat (bfloat 0xR3DB5)
  %1443 = fmul <2 x bfloat> %1417, splat (bfloat 0xR3DB5)
  %1444 = fmul <2 x bfloat> %1423, splat (bfloat 0xR3DB5)
  %1445 = fmul <2 x bfloat> %1429, splat (bfloat 0xR3DB5)
  %1446 = fmul <2 x bfloat> %1435, splat (bfloat 0xR3DB5)
  %1447 = fmul <2 x bfloat> %1394, splat (bfloat 0xR3DB5)
  %1448 = fmul <2 x bfloat> %1400, splat (bfloat 0xR3DB5)
  %1449 = fmul <2 x bfloat> %1406, splat (bfloat 0xR3DB5)
  %1450 = fmul <2 x bfloat> %1412, splat (bfloat 0xR3DB5)
  %1451 = fmul <2 x bfloat> %1418, splat (bfloat 0xR3DB5)
  %1452 = fmul <2 x bfloat> %1424, splat (bfloat 0xR3DB5)
  %1453 = fmul <2 x bfloat> %1430, splat (bfloat 0xR3DB5)
  %1454 = fmul <2 x bfloat> %1436, splat (bfloat 0xR3DB5)
  %1455 = fmul <2 x bfloat> %1395, splat (bfloat 0xR3DB5)
  %1456 = fmul <2 x bfloat> %1401, splat (bfloat 0xR3DB5)
  %1457 = fmul <2 x bfloat> %1407, splat (bfloat 0xR3DB5)
  %1458 = fmul <2 x bfloat> %1413, splat (bfloat 0xR3DB5)
  %1459 = fmul <2 x bfloat> %1419, splat (bfloat 0xR3DB5)
  %1460 = fmul <2 x bfloat> %1425, splat (bfloat 0xR3DB5)
  %1461 = fmul <2 x bfloat> %1431, splat (bfloat 0xR3DB5)
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  store <4 x i32> %1278, ptr addrspace(3) %1099, align 16
  store <4 x i32> %1294, ptr addrspace(3) %1116, align 16
  store <4 x i32> %1310, ptr addrspace(3) %1135, align 16
  store <4 x i32> %1326, ptr addrspace(3) %1152, align 16
  store <4 x i32> %1342, ptr addrspace(3) %1171, align 16
  store <4 x i32> %1358, ptr addrspace(3) %1188, align 16
  store <4 x i32> %1374, ptr addrspace(3) %1207, align 16
  store <4 x i32> %1390, ptr addrspace(3) %1224, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %1462 = fmul <2 x bfloat> %1437, splat (bfloat 0xR3DB5)
  %1463 = load <8 x bfloat>, ptr addrspace(3) %1249, align 16
  %1464 = shufflevector <8 x bfloat> %1463, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1465 = shufflevector <8 x bfloat> %1463, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1466 = shufflevector <8 x bfloat> %1463, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1467 = shufflevector <8 x bfloat> %1463, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1468 = fmul <2 x bfloat> %1464, splat (bfloat 0xR3DB5)
  %1469 = load <8 x bfloat>, ptr addrspace(3) %1253, align 16
  %1470 = shufflevector <8 x bfloat> %1469, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1471 = shufflevector <8 x bfloat> %1469, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1472 = shufflevector <8 x bfloat> %1469, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1473 = shufflevector <8 x bfloat> %1469, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1474 = fmul <2 x bfloat> %1470, splat (bfloat 0xR3DB5)
  %1475 = load <8 x bfloat>, ptr addrspace(3) %1257, align 16
  %1476 = shufflevector <8 x bfloat> %1475, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1477 = shufflevector <8 x bfloat> %1475, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1478 = shufflevector <8 x bfloat> %1475, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1479 = shufflevector <8 x bfloat> %1475, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1480 = fmul <2 x bfloat> %1476, splat (bfloat 0xR3DB5)
  %1481 = load <8 x bfloat>, ptr addrspace(3) %1261, align 16
  %1482 = shufflevector <8 x bfloat> %1481, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1483 = shufflevector <8 x bfloat> %1481, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1484 = shufflevector <8 x bfloat> %1481, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1485 = shufflevector <8 x bfloat> %1481, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1486 = fmul <2 x bfloat> %1482, splat (bfloat 0xR3DB5)
  %1487 = load <8 x bfloat>, ptr addrspace(3) %1250, align 16
  %1488 = shufflevector <8 x bfloat> %1487, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1489 = shufflevector <8 x bfloat> %1487, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1490 = shufflevector <8 x bfloat> %1487, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1491 = shufflevector <8 x bfloat> %1487, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1492 = fmul <2 x bfloat> %1488, splat (bfloat 0xR3DB5)
  %1493 = load <8 x bfloat>, ptr addrspace(3) %1254, align 16
  %1494 = shufflevector <8 x bfloat> %1493, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1495 = shufflevector <8 x bfloat> %1493, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1496 = shufflevector <8 x bfloat> %1493, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1497 = shufflevector <8 x bfloat> %1493, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1498 = fmul <2 x bfloat> %1494, splat (bfloat 0xR3DB5)
  %1499 = load <8 x bfloat>, ptr addrspace(3) %1258, align 16
  %1500 = shufflevector <8 x bfloat> %1499, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1501 = shufflevector <8 x bfloat> %1499, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1502 = shufflevector <8 x bfloat> %1499, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1503 = shufflevector <8 x bfloat> %1499, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1504 = fmul <2 x bfloat> %1500, splat (bfloat 0xR3DB5)
  %1505 = load <8 x bfloat>, ptr addrspace(3) %1262, align 16
  %1506 = shufflevector <8 x bfloat> %1505, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1507 = shufflevector <8 x bfloat> %1505, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1508 = shufflevector <8 x bfloat> %1505, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1509 = shufflevector <8 x bfloat> %1505, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1510 = fmul <2 x bfloat> %1506, splat (bfloat 0xR3DB5)
  %1511 = fmul <2 x bfloat> %1465, splat (bfloat 0xR3DB5)
  %1512 = fmul <2 x bfloat> %1471, splat (bfloat 0xR3DB5)
  %1513 = fmul <2 x bfloat> %1477, splat (bfloat 0xR3DB5)
  %1514 = fmul <2 x bfloat> %1483, splat (bfloat 0xR3DB5)
  %1515 = fmul <2 x bfloat> %1489, splat (bfloat 0xR3DB5)
  %1516 = fmul <2 x bfloat> %1495, splat (bfloat 0xR3DB5)
  %1517 = fmul <2 x bfloat> %1501, splat (bfloat 0xR3DB5)
  %1518 = fmul <2 x bfloat> %1507, splat (bfloat 0xR3DB5)
  %1519 = fmul <2 x bfloat> %1466, splat (bfloat 0xR3DB5)
  %1520 = fmul <2 x bfloat> %1472, splat (bfloat 0xR3DB5)
  %1521 = fmul <2 x bfloat> %1478, splat (bfloat 0xR3DB5)
  %1522 = fmul <2 x bfloat> %1484, splat (bfloat 0xR3DB5)
  %1523 = fmul <2 x bfloat> %1490, splat (bfloat 0xR3DB5)
  %1524 = fmul <2 x bfloat> %1496, splat (bfloat 0xR3DB5)
  %1525 = fmul <2 x bfloat> %1502, splat (bfloat 0xR3DB5)
  %1526 = fmul <2 x bfloat> %1508, splat (bfloat 0xR3DB5)
  %1527 = fmul <2 x bfloat> %1467, splat (bfloat 0xR3DB5)
  %1528 = fmul <2 x bfloat> %1473, splat (bfloat 0xR3DB5)
  %1529 = fmul <2 x bfloat> %1479, splat (bfloat 0xR3DB5)
  %1530 = fmul <2 x bfloat> %1485, splat (bfloat 0xR3DB5)
  %1531 = fmul <2 x bfloat> %1491, splat (bfloat 0xR3DB5)
  %1532 = fmul <2 x bfloat> %1497, splat (bfloat 0xR3DB5)
  %1533 = fmul <2 x bfloat> %1503, splat (bfloat 0xR3DB5)
  %1534 = fmul <2 x bfloat> %1509, splat (bfloat 0xR3DB5)
  %1535 = getelementptr [2 x i8], ptr addrspace(1) %3, i64 %19
  %.idx = shl nuw nsw i64 %8, 19
  %1536 = getelementptr i8, ptr addrspace(1) %1535, i64 %.idx
  %.idx318 = shl nuw nsw i64 %6, 25
  %1537 = getelementptr i8, ptr addrspace(1) %1536, i64 %.idx318
  %1538 = getelementptr [2 x i8], ptr addrspace(1) %1537, i64 %25
  %1539 = getelementptr i8, ptr addrspace(1) %1538, i64 32768
  %1540 = getelementptr i8, ptr addrspace(1) %1538, i64 65536
  %1541 = getelementptr i8, ptr addrspace(1) %1538, i64 98304
  %1542 = getelementptr i8, ptr addrspace(1) %1538, i64 131072
  %1543 = getelementptr i8, ptr addrspace(1) %1538, i64 163840
  %1544 = getelementptr i8, ptr addrspace(1) %1538, i64 196608
  %1545 = getelementptr i8, ptr addrspace(1) %1538, i64 229376
  %1546 = getelementptr i8, ptr addrspace(1) %1538, i64 262144
  %1547 = getelementptr i8, ptr addrspace(1) %1538, i64 294912
  %1548 = getelementptr i8, ptr addrspace(1) %1538, i64 327680
  %1549 = getelementptr i8, ptr addrspace(1) %1538, i64 360448
  %1550 = getelementptr i8, ptr addrspace(1) %1538, i64 393216
  %1551 = getelementptr i8, ptr addrspace(1) %1538, i64 425984
  %1552 = getelementptr i8, ptr addrspace(1) %1538, i64 458752
  %1553 = getelementptr i8, ptr addrspace(1) %1538, i64 491520
  %1554 = bitcast <2 x bfloat> %1396 to i32
  %1555 = bitcast <2 x bfloat> %1402 to i32
  %1556 = bitcast <2 x bfloat> %1408 to i32
  %1557 = bitcast <2 x bfloat> %1414 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1554, i32 %1555, i32 %1556, i32 %1557, ptr addrspace(1) %1538) #7
  %1558 = bitcast <2 x bfloat> %1420 to i32
  %1559 = bitcast <2 x bfloat> %1426 to i32
  %1560 = bitcast <2 x bfloat> %1432 to i32
  %1561 = bitcast <2 x bfloat> %1438 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1558, i32 %1559, i32 %1560, i32 %1561, ptr addrspace(1) %1539) #7
  %1562 = bitcast <2 x bfloat> %1439 to i32
  %1563 = bitcast <2 x bfloat> %1440 to i32
  %1564 = bitcast <2 x bfloat> %1441 to i32
  %1565 = bitcast <2 x bfloat> %1442 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1562, i32 %1563, i32 %1564, i32 %1565, ptr addrspace(1) %1540) #7
  %1566 = bitcast <2 x bfloat> %1443 to i32
  %1567 = bitcast <2 x bfloat> %1444 to i32
  %1568 = bitcast <2 x bfloat> %1445 to i32
  %1569 = bitcast <2 x bfloat> %1446 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1566, i32 %1567, i32 %1568, i32 %1569, ptr addrspace(1) %1541) #7
  %1570 = bitcast <2 x bfloat> %1447 to i32
  %1571 = bitcast <2 x bfloat> %1448 to i32
  %1572 = bitcast <2 x bfloat> %1449 to i32
  %1573 = bitcast <2 x bfloat> %1450 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1570, i32 %1571, i32 %1572, i32 %1573, ptr addrspace(1) %1542) #7
  %1574 = bitcast <2 x bfloat> %1451 to i32
  %1575 = bitcast <2 x bfloat> %1452 to i32
  %1576 = bitcast <2 x bfloat> %1453 to i32
  %1577 = bitcast <2 x bfloat> %1454 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1574, i32 %1575, i32 %1576, i32 %1577, ptr addrspace(1) %1543) #7
  %1578 = bitcast <2 x bfloat> %1455 to i32
  %1579 = bitcast <2 x bfloat> %1456 to i32
  %1580 = bitcast <2 x bfloat> %1457 to i32
  %1581 = bitcast <2 x bfloat> %1458 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1578, i32 %1579, i32 %1580, i32 %1581, ptr addrspace(1) %1544) #7
  %1582 = bitcast <2 x bfloat> %1459 to i32
  %1583 = bitcast <2 x bfloat> %1460 to i32
  %1584 = bitcast <2 x bfloat> %1461 to i32
  %1585 = bitcast <2 x bfloat> %1462 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1582, i32 %1583, i32 %1584, i32 %1585, ptr addrspace(1) %1545) #7
  %1586 = bitcast <2 x bfloat> %1468 to i32
  %1587 = bitcast <2 x bfloat> %1474 to i32
  %1588 = bitcast <2 x bfloat> %1480 to i32
  %1589 = bitcast <2 x bfloat> %1486 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1586, i32 %1587, i32 %1588, i32 %1589, ptr addrspace(1) %1546) #7
  %1590 = bitcast <2 x bfloat> %1492 to i32
  %1591 = bitcast <2 x bfloat> %1498 to i32
  %1592 = bitcast <2 x bfloat> %1504 to i32
  %1593 = bitcast <2 x bfloat> %1510 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1590, i32 %1591, i32 %1592, i32 %1593, ptr addrspace(1) %1547) #7
  %1594 = bitcast <2 x bfloat> %1511 to i32
  %1595 = bitcast <2 x bfloat> %1512 to i32
  %1596 = bitcast <2 x bfloat> %1513 to i32
  %1597 = bitcast <2 x bfloat> %1514 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1594, i32 %1595, i32 %1596, i32 %1597, ptr addrspace(1) %1548) #7
  %1598 = bitcast <2 x bfloat> %1515 to i32
  %1599 = bitcast <2 x bfloat> %1516 to i32
  %1600 = bitcast <2 x bfloat> %1517 to i32
  %1601 = bitcast <2 x bfloat> %1518 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1598, i32 %1599, i32 %1600, i32 %1601, ptr addrspace(1) %1549) #7
  %1602 = bitcast <2 x bfloat> %1519 to i32
  %1603 = bitcast <2 x bfloat> %1520 to i32
  %1604 = bitcast <2 x bfloat> %1521 to i32
  %1605 = bitcast <2 x bfloat> %1522 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1602, i32 %1603, i32 %1604, i32 %1605, ptr addrspace(1) %1550) #7
  %1606 = bitcast <2 x bfloat> %1523 to i32
  %1607 = bitcast <2 x bfloat> %1524 to i32
  %1608 = bitcast <2 x bfloat> %1525 to i32
  %1609 = bitcast <2 x bfloat> %1526 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1606, i32 %1607, i32 %1608, i32 %1609, ptr addrspace(1) %1551) #7
  %1610 = bitcast <2 x bfloat> %1527 to i32
  %1611 = bitcast <2 x bfloat> %1528 to i32
  %1612 = bitcast <2 x bfloat> %1529 to i32
  %1613 = bitcast <2 x bfloat> %1530 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1610, i32 %1611, i32 %1612, i32 %1613, ptr addrspace(1) %1552) #7
  %1614 = bitcast <2 x bfloat> %1531 to i32
  %1615 = bitcast <2 x bfloat> %1532 to i32
  %1616 = bitcast <2 x bfloat> %1533 to i32
  %1617 = bitcast <2 x bfloat> %1534 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %1614, i32 %1615, i32 %1616, i32 %1617, ptr addrspace(1) %1553) #7
  ret void
}

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.commit.group() #7

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.wait.group(i32 immarg) #7

; Function Attrs: convergent nocallback nofree nounwind memory(argmem: read)
declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) readonly captures(none)) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define ptx_kernel void @loop_broadcast_fusion_1(ptr noalias writeonly align 256 captures(none) dereferenceable(1073741824) %0) local_unnamed_addr #8 {
  %2 = addrspacecast ptr %0 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !9
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %5 = shl nuw nsw i32 %4, 2
  %6 = shl nuw nsw i32 %3, 9
  %7 = or disjoint i32 %5, %6
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds [2 x i8], ptr addrspace(1) %2, i64 %8
  store <4 x bfloat> splat (bfloat 0xRCE6E), ptr addrspace(1) %9, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define ptx_kernel void @wrapped_iota(ptr noalias writeonly align 256 captures(none) dereferenceable(67108864) %0) local_unnamed_addr #8 {
  %2 = addrspacecast ptr %0 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %5 = lshr i32 %3, 3
  %6 = shl nuw nsw i32 %4, 2
  %7 = shl nuw nsw i32 %3, 9
  %8 = or disjoint i32 %6, %7
  %9 = insertelement <4 x i32> poison, i32 %5, i64 0
  %10 = shufflevector <4 x i32> %9, <4 x i32> poison, <4 x i32> zeroinitializer
  %11 = zext nneg i32 %8 to i64
  %12 = getelementptr inbounds [4 x i8], ptr addrspace(1) %2, i64 %11
  store <4 x i32> %10, ptr addrspace(1) %12, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define ptx_kernel void @wrapped_iota_1(ptr noalias writeonly align 256 captures(none) dereferenceable(67108864) %0) local_unnamed_addr #8 {
  %2 = addrspacecast ptr %0 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %5 = shl nuw nsw i32 %4, 2
  %6 = shl nuw nsw i32 %3, 9
  %7 = or disjoint i32 %5, %6
  %8 = and i32 %6, 3584
  %9 = or disjoint i32 %8, %5
  %10 = or disjoint i32 %9, 1
  %11 = or disjoint i32 %9, 2
  %12 = or disjoint i32 %9, 3
  %13 = insertelement <4 x i32> poison, i32 %9, i64 0
  %14 = insertelement <4 x i32> %13, i32 %10, i64 1
  %15 = insertelement <4 x i32> %14, i32 %11, i64 2
  %16 = insertelement <4 x i32> %15, i32 %12, i64 3
  %17 = zext nneg i32 %7 to i64
  %18 = getelementptr inbounds [4 x i8], ptr addrspace(1) %2, i64 %17
  store <4 x i32> %16, ptr addrspace(1) %18, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_compare(ptr noalias readonly align 256 captures(none) dereferenceable(67108864) %0, ptr noalias readonly align 256 captures(none) dereferenceable(67108864) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(16777216) %2) local_unnamed_addr #5 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %9 = shl nuw nsw i32 %8, 2
  %10 = shl nuw nsw i32 %7, 9
  %11 = or disjoint i32 %9, %10
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %12
  %14 = getelementptr inbounds [4 x i8], ptr addrspace(1) %5, i64 %12
  %15 = load <4 x i32>, ptr addrspace(1) %14, align 16, !invariant.load !6
  %16 = extractelement <4 x i32> %15, i32 0
  %17 = extractelement <4 x i32> %15, i32 1
  %18 = extractelement <4 x i32> %15, i32 2
  %19 = extractelement <4 x i32> %15, i32 3
  %20 = load <4 x i32>, ptr addrspace(1) %13, align 16, !invariant.load !6
  %21 = extractelement <4 x i32> %20, i32 0
  %22 = extractelement <4 x i32> %20, i32 1
  %23 = extractelement <4 x i32> %20, i32 2
  %24 = extractelement <4 x i32> %20, i32 3
  %25 = icmp sge i32 %16, %21
  %26 = zext i1 %25 to i8
  %27 = icmp sge i32 %17, %22
  %28 = zext i1 %27 to i8
  %29 = icmp sge i32 %18, %23
  %30 = zext i1 %29 to i8
  %31 = icmp sge i32 %19, %24
  %32 = zext i1 %31 to i8
  %33 = insertelement <4 x i8> poison, i8 %26, i64 0
  %34 = insertelement <4 x i8> %33, i8 %28, i64 1
  %35 = insertelement <4 x i8> %34, i8 %30, i64 2
  %36 = insertelement <4 x i8> %35, i8 %32, i64 3
  %37 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 %12
  store <4 x i8> %36, ptr addrspace(1) %37, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_broadcast(ptr noalias readonly align 256 captures(none) dereferenceable(16777216) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(536870912) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !9
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = and i32 %8, 16776704
  %11 = or disjoint i32 %10, %7
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds i8, ptr addrspace(1) %3, i64 %12
  %14 = load <4 x i8>, ptr addrspace(1) %13, align 4, !invariant.load !6
  %15 = zext nneg i32 %9 to i64
  %16 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %15
  store <4 x i8> %14, ptr addrspace(1) %16, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_select(ptr noalias readonly align 256 captures(none) dereferenceable(536870912) %0, ptr noalias align 256 captures(none) dereferenceable(1073741824) %1, ptr noalias readonly align 256 captures(none) dereferenceable(1073741824) %2, ptr noalias readnone align 256 captures(none) dereferenceable(1073741824) %3) local_unnamed_addr #5 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !9
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %10 = shl nuw nsw i32 %9, 2
  %11 = shl nuw nsw i32 %8, 9
  %12 = or disjoint i32 %10, %11
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds [2 x i8], ptr addrspace(1) %5, i64 %13
  %15 = getelementptr inbounds [2 x i8], ptr addrspace(1) %6, i64 %13
  %16 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 %13
  %17 = load <32 x i1>, ptr addrspace(1) %16, align 4, !invariant.load !6
  %18 = load <4 x bfloat>, ptr addrspace(1) %15, align 8
  %19 = extractelement <4 x bfloat> %18, i32 0
  %20 = extractelement <4 x bfloat> %18, i32 1
  %21 = extractelement <4 x bfloat> %18, i32 2
  %22 = extractelement <4 x bfloat> %18, i32 3
  %23 = load <4 x bfloat>, ptr addrspace(1) %14, align 8, !invariant.load !6
  %24 = extractelement <4 x bfloat> %23, i32 0
  %25 = extractelement <4 x bfloat> %23, i32 1
  %26 = extractelement <4 x bfloat> %23, i32 2
  %27 = extractelement <4 x bfloat> %23, i32 3
  %28 = extractelement <32 x i1> %17, i64 0
  %29 = select i1 %28, bfloat %19, bfloat %24
  %30 = extractelement <32 x i1> %17, i64 8
  %31 = select i1 %30, bfloat %20, bfloat %25
  %32 = extractelement <32 x i1> %17, i64 16
  %33 = select i1 %32, bfloat %21, bfloat %26
  %34 = extractelement <32 x i1> %17, i64 24
  %35 = select i1 %34, bfloat %22, bfloat %27
  %36 = insertelement <4 x bfloat> poison, bfloat %29, i64 0
  %37 = insertelement <4 x bfloat> %36, bfloat %31, i64 1
  %38 = insertelement <4 x bfloat> %37, bfloat %33, i64 2
  %39 = insertelement <4 x bfloat> %38, bfloat %35, i64 3
  store <4 x bfloat> %39, ptr addrspace(1) %15, align 8
  ret void
}

; Function Attrs: nounwind
define ptx_kernel void @triton_softmax_4(ptr noalias align 256 dereferenceable(1073741824) %arg0, ptr noalias align 256 dereferenceable(1073741824) %arg1) local_unnamed_addr #4 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = zext i32 %3 to i64
  %5 = tail call range(i32 0, 128) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = shl i32 %5, 3
  %7 = zext i32 %6 to i64
  %8 = shl i64 %4, 12
  %9 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %8
  %10 = getelementptr [2 x i8], ptr addrspace(1) %9, i64 %7
  %11 = getelementptr i8, ptr addrspace(1) %10, i64 2048
  %12 = getelementptr i8, ptr addrspace(1) %10, i64 4096
  %13 = getelementptr i8, ptr addrspace(1) %10, i64 6144
  %14 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %10) #7
  %15 = extractvalue { i32, i32, i32, i32 } %14, 0
  %16 = bitcast i32 %15 to <2 x bfloat>
  %17 = extractvalue { i32, i32, i32, i32 } %14, 1
  %18 = bitcast i32 %17 to <2 x bfloat>
  %19 = extractvalue { i32, i32, i32, i32 } %14, 2
  %20 = bitcast i32 %19 to <2 x bfloat>
  %21 = extractvalue { i32, i32, i32, i32 } %14, 3
  %22 = bitcast i32 %21 to <2 x bfloat>
  %23 = extractelement <2 x bfloat> %16, i64 0
  %24 = extractelement <2 x bfloat> %16, i64 1
  %25 = extractelement <2 x bfloat> %18, i64 0
  %26 = extractelement <2 x bfloat> %18, i64 1
  %27 = extractelement <2 x bfloat> %20, i64 0
  %28 = extractelement <2 x bfloat> %20, i64 1
  %29 = extractelement <2 x bfloat> %22, i64 0
  %30 = extractelement <2 x bfloat> %22, i64 1
  %31 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %11) #7
  %32 = extractvalue { i32, i32, i32, i32 } %31, 0
  %33 = bitcast i32 %32 to <2 x bfloat>
  %34 = extractvalue { i32, i32, i32, i32 } %31, 1
  %35 = bitcast i32 %34 to <2 x bfloat>
  %36 = extractvalue { i32, i32, i32, i32 } %31, 2
  %37 = bitcast i32 %36 to <2 x bfloat>
  %38 = extractvalue { i32, i32, i32, i32 } %31, 3
  %39 = bitcast i32 %38 to <2 x bfloat>
  %40 = extractelement <2 x bfloat> %33, i64 0
  %41 = extractelement <2 x bfloat> %33, i64 1
  %42 = extractelement <2 x bfloat> %35, i64 0
  %43 = extractelement <2 x bfloat> %35, i64 1
  %44 = extractelement <2 x bfloat> %37, i64 0
  %45 = extractelement <2 x bfloat> %37, i64 1
  %46 = extractelement <2 x bfloat> %39, i64 0
  %47 = extractelement <2 x bfloat> %39, i64 1
  %48 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %12) #7
  %49 = extractvalue { i32, i32, i32, i32 } %48, 0
  %50 = bitcast i32 %49 to <2 x bfloat>
  %51 = extractvalue { i32, i32, i32, i32 } %48, 1
  %52 = bitcast i32 %51 to <2 x bfloat>
  %53 = extractvalue { i32, i32, i32, i32 } %48, 2
  %54 = bitcast i32 %53 to <2 x bfloat>
  %55 = extractvalue { i32, i32, i32, i32 } %48, 3
  %56 = bitcast i32 %55 to <2 x bfloat>
  %57 = extractelement <2 x bfloat> %50, i64 0
  %58 = extractelement <2 x bfloat> %50, i64 1
  %59 = extractelement <2 x bfloat> %52, i64 0
  %60 = extractelement <2 x bfloat> %52, i64 1
  %61 = extractelement <2 x bfloat> %54, i64 0
  %62 = extractelement <2 x bfloat> %54, i64 1
  %63 = extractelement <2 x bfloat> %56, i64 0
  %64 = extractelement <2 x bfloat> %56, i64 1
  %65 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %13) #7
  %66 = extractvalue { i32, i32, i32, i32 } %65, 0
  %67 = bitcast i32 %66 to <2 x bfloat>
  %68 = extractvalue { i32, i32, i32, i32 } %65, 1
  %69 = bitcast i32 %68 to <2 x bfloat>
  %70 = extractvalue { i32, i32, i32, i32 } %65, 2
  %71 = bitcast i32 %70 to <2 x bfloat>
  %72 = extractvalue { i32, i32, i32, i32 } %65, 3
  %73 = bitcast i32 %72 to <2 x bfloat>
  %74 = extractelement <2 x bfloat> %67, i64 0
  %75 = extractelement <2 x bfloat> %67, i64 1
  %76 = extractelement <2 x bfloat> %69, i64 0
  %77 = extractelement <2 x bfloat> %69, i64 1
  %78 = extractelement <2 x bfloat> %71, i64 0
  %79 = extractelement <2 x bfloat> %71, i64 1
  %80 = extractelement <2 x bfloat> %73, i64 0
  %81 = extractelement <2 x bfloat> %73, i64 1
  %82 = shl nuw nsw i32 %5, 4
  %83 = zext nneg i32 %82 to i64
  %84 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %83
  %85 = bitcast bfloat %41 to i16
  %86 = bitcast bfloat %43 to i16
  %87 = shufflevector <2 x bfloat> %16, <2 x bfloat> %33, <2 x i32> <i32 0, i32 2>
  %88 = bitcast <2 x bfloat> %87 to i32
  %89 = bitcast i32 %15 to <2 x i16>
  %90 = shufflevector <2 x i16> %89, <2 x i16> poison, <2 x i32> <i32 1, i32 poison>
  %91 = insertelement <2 x i16> %90, i16 %85, i64 1
  %92 = bitcast <2 x i16> %91 to i32
  %93 = shufflevector <2 x bfloat> %18, <2 x bfloat> %35, <2 x i32> <i32 0, i32 2>
  %94 = bitcast <2 x bfloat> %93 to i32
  %95 = bitcast i32 %17 to <2 x i16>
  %96 = shufflevector <2 x i16> %95, <2 x i16> poison, <2 x i32> <i32 1, i32 poison>
  %97 = insertelement <2 x i16> %96, i16 %86, i64 1
  %98 = bitcast <2 x i16> %97 to i32
  %99 = insertelement <4 x i32> poison, i32 %88, i64 0
  %100 = insertelement <4 x i32> %99, i32 %92, i64 1
  %101 = insertelement <4 x i32> %100, i32 %94, i64 2
  %102 = insertelement <4 x i32> %101, i32 %98, i64 3
  store <4 x i32> %102, ptr addrspace(3) %84, align 16
  %103 = xor i32 %82, 2112
  %104 = zext nneg i32 %103 to i64
  %105 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %104
  %106 = bitcast bfloat %45 to i16
  %107 = bitcast bfloat %47 to i16
  %108 = shufflevector <2 x bfloat> %20, <2 x bfloat> %37, <2 x i32> <i32 0, i32 2>
  %109 = bitcast <2 x bfloat> %108 to i32
  %110 = bitcast i32 %19 to <2 x i16>
  %111 = shufflevector <2 x i16> %110, <2 x i16> poison, <2 x i32> <i32 1, i32 poison>
  %112 = insertelement <2 x i16> %111, i16 %106, i64 1
  %113 = bitcast <2 x i16> %112 to i32
  %114 = shufflevector <2 x bfloat> %22, <2 x bfloat> %39, <2 x i32> <i32 0, i32 2>
  %115 = bitcast <2 x bfloat> %114 to i32
  %116 = bitcast i32 %21 to <2 x i16>
  %117 = shufflevector <2 x i16> %116, <2 x i16> poison, <2 x i32> <i32 1, i32 poison>
  %118 = insertelement <2 x i16> %117, i16 %107, i64 1
  %119 = bitcast <2 x i16> %118 to i32
  %120 = insertelement <4 x i32> poison, i32 %109, i64 0
  %121 = insertelement <4 x i32> %120, i32 %113, i64 1
  %122 = insertelement <4 x i32> %121, i32 %115, i64 2
  %123 = insertelement <4 x i32> %122, i32 %119, i64 3
  store <4 x i32> %123, ptr addrspace(3) %105, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %124 = shl nuw nsw i32 %5, 5
  %125 = and i32 %124, 768
  %126 = and i32 %6, 48
  %127 = and i32 %5, 96
  %128 = shl nuw nsw i32 %127, 1
  %129 = and i32 %5, 1
  %130 = icmp eq i32 %129, 0
  %131 = select i1 %130, i32 0, i32 2112
  %132 = or disjoint i32 %125, %126
  %133 = xor i32 %131, %128
  %134 = or disjoint i32 %132, %133
  %135 = zext nneg i32 %134 to i64
  %136 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %135
  %137 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %136)
  %138 = extractvalue { i32, i32, i32, i32 } %137, 0
  %139 = bitcast i32 %138 to <2 x bfloat>
  %140 = extractelement <2 x bfloat> %139, i64 0
  %141 = extractelement <2 x bfloat> %139, i64 1
  %142 = extractvalue { i32, i32, i32, i32 } %137, 1
  %143 = bitcast i32 %142 to <2 x bfloat>
  %144 = extractelement <2 x bfloat> %143, i64 0
  %145 = extractelement <2 x bfloat> %143, i64 1
  %146 = extractvalue { i32, i32, i32, i32 } %137, 2
  %147 = bitcast i32 %146 to <2 x bfloat>
  %148 = extractelement <2 x bfloat> %147, i64 0
  %149 = extractelement <2 x bfloat> %147, i64 1
  %150 = extractvalue { i32, i32, i32, i32 } %137, 3
  %151 = bitcast i32 %150 to <2 x bfloat>
  %152 = extractelement <2 x bfloat> %151, i64 0
  %153 = extractelement <2 x bfloat> %151, i64 1
  %154 = getelementptr inbounds nuw i8, ptr addrspace(3) %136, i64 1024
  %155 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %154)
  %156 = extractvalue { i32, i32, i32, i32 } %155, 0
  %157 = bitcast i32 %156 to <2 x bfloat>
  %158 = extractelement <2 x bfloat> %157, i64 0
  %159 = extractelement <2 x bfloat> %157, i64 1
  %160 = extractvalue { i32, i32, i32, i32 } %155, 1
  %161 = bitcast i32 %160 to <2 x bfloat>
  %162 = extractelement <2 x bfloat> %161, i64 0
  %163 = extractelement <2 x bfloat> %161, i64 1
  %164 = extractvalue { i32, i32, i32, i32 } %155, 2
  %165 = bitcast i32 %164 to <2 x bfloat>
  %166 = extractelement <2 x bfloat> %165, i64 0
  %167 = extractelement <2 x bfloat> %165, i64 1
  %168 = extractvalue { i32, i32, i32, i32 } %155, 3
  %169 = bitcast i32 %168 to <2 x bfloat>
  %170 = extractelement <2 x bfloat> %169, i64 0
  %171 = extractelement <2 x bfloat> %169, i64 1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %172 = bitcast bfloat %75 to i16
  %173 = bitcast bfloat %77 to i16
  %174 = shufflevector <2 x bfloat> %50, <2 x bfloat> %67, <2 x i32> <i32 0, i32 2>
  %175 = bitcast <2 x bfloat> %174 to i32
  %176 = bitcast i32 %49 to <2 x i16>
  %177 = shufflevector <2 x i16> %176, <2 x i16> poison, <2 x i32> <i32 1, i32 poison>
  %178 = insertelement <2 x i16> %177, i16 %172, i64 1
  %179 = bitcast <2 x i16> %178 to i32
  %180 = shufflevector <2 x bfloat> %52, <2 x bfloat> %69, <2 x i32> <i32 0, i32 2>
  %181 = bitcast <2 x bfloat> %180 to i32
  %182 = bitcast i32 %51 to <2 x i16>
  %183 = shufflevector <2 x i16> %182, <2 x i16> poison, <2 x i32> <i32 1, i32 poison>
  %184 = insertelement <2 x i16> %183, i16 %173, i64 1
  %185 = bitcast <2 x i16> %184 to i32
  %186 = insertelement <4 x i32> poison, i32 %175, i64 0
  %187 = insertelement <4 x i32> %186, i32 %179, i64 1
  %188 = insertelement <4 x i32> %187, i32 %181, i64 2
  %189 = insertelement <4 x i32> %188, i32 %185, i64 3
  store <4 x i32> %189, ptr addrspace(3) %84, align 16
  %190 = bitcast bfloat %79 to i16
  %191 = bitcast bfloat %81 to i16
  %192 = shufflevector <2 x bfloat> %54, <2 x bfloat> %71, <2 x i32> <i32 0, i32 2>
  %193 = bitcast <2 x bfloat> %192 to i32
  %194 = bitcast i32 %53 to <2 x i16>
  %195 = shufflevector <2 x i16> %194, <2 x i16> poison, <2 x i32> <i32 1, i32 poison>
  %196 = insertelement <2 x i16> %195, i16 %190, i64 1
  %197 = bitcast <2 x i16> %196 to i32
  %198 = shufflevector <2 x bfloat> %56, <2 x bfloat> %73, <2 x i32> <i32 0, i32 2>
  %199 = bitcast <2 x bfloat> %198 to i32
  %200 = bitcast i32 %55 to <2 x i16>
  %201 = shufflevector <2 x i16> %200, <2 x i16> poison, <2 x i32> <i32 1, i32 poison>
  %202 = insertelement <2 x i16> %201, i16 %191, i64 1
  %203 = bitcast <2 x i16> %202 to i32
  %204 = insertelement <4 x i32> poison, i32 %193, i64 0
  %205 = insertelement <4 x i32> %204, i32 %197, i64 1
  %206 = insertelement <4 x i32> %205, i32 %199, i64 2
  %207 = insertelement <4 x i32> %206, i32 %203, i64 3
  store <4 x i32> %207, ptr addrspace(3) %105, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %208 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %136)
  %209 = extractvalue { i32, i32, i32, i32 } %208, 0
  %210 = bitcast i32 %209 to <2 x bfloat>
  %211 = extractelement <2 x bfloat> %210, i64 0
  %212 = extractelement <2 x bfloat> %210, i64 1
  %213 = extractvalue { i32, i32, i32, i32 } %208, 1
  %214 = bitcast i32 %213 to <2 x bfloat>
  %215 = extractelement <2 x bfloat> %214, i64 0
  %216 = extractelement <2 x bfloat> %214, i64 1
  %217 = extractvalue { i32, i32, i32, i32 } %208, 2
  %218 = bitcast i32 %217 to <2 x bfloat>
  %219 = extractelement <2 x bfloat> %218, i64 0
  %220 = extractelement <2 x bfloat> %218, i64 1
  %221 = extractvalue { i32, i32, i32, i32 } %208, 3
  %222 = bitcast i32 %221 to <2 x bfloat>
  %223 = extractelement <2 x bfloat> %222, i64 0
  %224 = extractelement <2 x bfloat> %222, i64 1
  %225 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %154)
  %226 = extractvalue { i32, i32, i32, i32 } %225, 0
  %227 = bitcast i32 %226 to <2 x bfloat>
  %228 = extractelement <2 x bfloat> %227, i64 0
  %229 = extractelement <2 x bfloat> %227, i64 1
  %230 = extractvalue { i32, i32, i32, i32 } %225, 1
  %231 = bitcast i32 %230 to <2 x bfloat>
  %232 = extractelement <2 x bfloat> %231, i64 0
  %233 = extractelement <2 x bfloat> %231, i64 1
  %234 = extractvalue { i32, i32, i32, i32 } %225, 2
  %235 = bitcast i32 %234 to <2 x bfloat>
  %236 = extractelement <2 x bfloat> %235, i64 0
  %237 = extractelement <2 x bfloat> %235, i64 1
  %238 = extractvalue { i32, i32, i32, i32 } %225, 3
  %239 = bitcast i32 %238 to <2 x bfloat>
  %240 = extractelement <2 x bfloat> %239, i64 0
  %241 = extractelement <2 x bfloat> %239, i64 1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %242 = tail call bfloat @llvm.maximum.bf16(bfloat %140, bfloat %144)
  %243 = tail call bfloat @llvm.maximum.bf16(bfloat %148, bfloat %152)
  %244 = tail call bfloat @llvm.maximum.bf16(bfloat %158, bfloat %162)
  %245 = tail call bfloat @llvm.maximum.bf16(bfloat %166, bfloat %170)
  %246 = tail call bfloat @llvm.maximum.bf16(bfloat %141, bfloat %145)
  %247 = tail call bfloat @llvm.maximum.bf16(bfloat %149, bfloat %153)
  %248 = tail call bfloat @llvm.maximum.bf16(bfloat %159, bfloat %163)
  %249 = tail call bfloat @llvm.maximum.bf16(bfloat %167, bfloat %171)
  %250 = tail call bfloat @llvm.maximum.bf16(bfloat %211, bfloat %215)
  %251 = tail call bfloat @llvm.maximum.bf16(bfloat %219, bfloat %223)
  %252 = tail call bfloat @llvm.maximum.bf16(bfloat %228, bfloat %232)
  %253 = tail call bfloat @llvm.maximum.bf16(bfloat %236, bfloat %240)
  %254 = tail call bfloat @llvm.maximum.bf16(bfloat %212, bfloat %216)
  %255 = tail call bfloat @llvm.maximum.bf16(bfloat %220, bfloat %224)
  %256 = tail call bfloat @llvm.maximum.bf16(bfloat %229, bfloat %233)
  %257 = tail call bfloat @llvm.maximum.bf16(bfloat %237, bfloat %241)
  %258 = tail call bfloat @llvm.maximum.bf16(bfloat %242, bfloat %243)
  %259 = tail call bfloat @llvm.maximum.bf16(bfloat %244, bfloat %245)
  %260 = tail call bfloat @llvm.maximum.bf16(bfloat %246, bfloat %247)
  %261 = tail call bfloat @llvm.maximum.bf16(bfloat %248, bfloat %249)
  %262 = tail call bfloat @llvm.maximum.bf16(bfloat %250, bfloat %251)
  %263 = tail call bfloat @llvm.maximum.bf16(bfloat %252, bfloat %253)
  %264 = tail call bfloat @llvm.maximum.bf16(bfloat %254, bfloat %255)
  %265 = tail call bfloat @llvm.maximum.bf16(bfloat %256, bfloat %257)
  %266 = tail call bfloat @llvm.maximum.bf16(bfloat %258, bfloat %259)
  %267 = tail call bfloat @llvm.maximum.bf16(bfloat %260, bfloat %261)
  %268 = tail call bfloat @llvm.maximum.bf16(bfloat %262, bfloat %263)
  %269 = tail call bfloat @llvm.maximum.bf16(bfloat %264, bfloat %265)
  %270 = tail call bfloat @llvm.maximum.bf16(bfloat %266, bfloat %267)
  %271 = tail call bfloat @llvm.maximum.bf16(bfloat %268, bfloat %269)
  %272 = tail call bfloat @llvm.maximum.bf16(bfloat %270, bfloat %271)
  %273 = bitcast bfloat %272 to i16
  %274 = zext i16 %273 to i32
  %275 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %274, i32 16, i32 31)
  %276 = trunc i32 %275 to i16
  %277 = bitcast i16 %276 to bfloat
  %278 = tail call bfloat @llvm.maximum.bf16(bfloat %272, bfloat %277)
  %279 = bitcast bfloat %278 to i16
  %280 = zext i16 %279 to i32
  %281 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %280, i32 8, i32 31)
  %282 = trunc i32 %281 to i16
  %283 = bitcast i16 %282 to bfloat
  %284 = tail call bfloat @llvm.maximum.bf16(bfloat %278, bfloat %283)
  %285 = bitcast bfloat %284 to i16
  %286 = zext i16 %285 to i32
  %287 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %286, i32 4, i32 31)
  %288 = trunc i32 %287 to i16
  %289 = bitcast i16 %288 to bfloat
  %290 = tail call bfloat @llvm.maximum.bf16(bfloat %284, bfloat %289)
  %291 = bitcast bfloat %290 to i16
  %292 = zext i16 %291 to i32
  %293 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %292, i32 2, i32 31)
  %294 = trunc i32 %293 to i16
  %295 = bitcast i16 %294 to bfloat
  %296 = tail call bfloat @llvm.maximum.bf16(bfloat %290, bfloat %295)
  %297 = bitcast bfloat %296 to i16
  %298 = zext i16 %297 to i32
  %299 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %298, i32 1, i32 31)
  %300 = trunc i32 %299 to i16
  %301 = bitcast i16 %300 to bfloat
  %302 = tail call bfloat @llvm.maximum.bf16(bfloat %296, bfloat %301)
  %303 = lshr exact i32 %127, 4
  %304 = zext nneg i32 %303 to i64
  %305 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %304
  store bfloat %302, ptr addrspace(3) %305, align 2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %306 = shl nuw nsw i32 %5, 1
  %307 = and i32 %306, 6
  %308 = zext nneg i32 %307 to i64
  %309 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %308
  %310 = load i16, ptr addrspace(3) %309, align 2
  %311 = bitcast i16 %310 to bfloat
  %312 = zext i16 %310 to i32
  %313 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %312, i32 2, i32 31)
  %314 = trunc i32 %313 to i16
  %315 = bitcast i16 %314 to bfloat
  %316 = tail call bfloat @llvm.maximum.bf16(bfloat %311, bfloat %315)
  %317 = bitcast bfloat %316 to i16
  %318 = zext i16 %317 to i32
  %319 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %318, i32 1, i32 31)
  %320 = trunc i32 %319 to i16
  %321 = bitcast i16 %320 to bfloat
  %322 = tail call bfloat @llvm.maximum.bf16(bfloat %316, bfloat %321)
  %323 = fsub bfloat %23, %322
  %324 = fsub bfloat %24, %322
  %325 = fsub bfloat %25, %322
  %326 = fsub bfloat %26, %322
  %327 = fsub bfloat %27, %322
  %328 = fsub bfloat %28, %322
  %329 = fsub bfloat %29, %322
  %330 = fsub bfloat %30, %322
  %331 = fsub bfloat %40, %322
  %332 = fsub bfloat %41, %322
  %333 = fsub bfloat %42, %322
  %334 = fsub bfloat %43, %322
  %335 = fsub bfloat %44, %322
  %336 = fsub bfloat %45, %322
  %337 = fsub bfloat %46, %322
  %338 = fsub bfloat %47, %322
  %339 = fsub bfloat %57, %322
  %340 = fsub bfloat %58, %322
  %341 = fsub bfloat %59, %322
  %342 = fsub bfloat %60, %322
  %343 = fsub bfloat %61, %322
  %344 = fsub bfloat %62, %322
  %345 = fsub bfloat %63, %322
  %346 = fsub bfloat %64, %322
  %347 = fsub bfloat %74, %322
  %348 = fsub bfloat %75, %322
  %349 = fsub bfloat %76, %322
  %350 = fsub bfloat %77, %322
  %351 = fsub bfloat %78, %322
  %352 = fsub bfloat %79, %322
  %353 = fsub bfloat %80, %322
  %354 = fsub bfloat %81, %322
  %355 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %8
  %356 = getelementptr [2 x i8], ptr addrspace(1) %355, i64 %7
  %357 = getelementptr i8, ptr addrspace(1) %356, i64 2048
  %358 = getelementptr i8, ptr addrspace(1) %356, i64 4096
  %359 = getelementptr i8, ptr addrspace(1) %356, i64 6144
  %360 = insertelement <2 x bfloat> poison, bfloat %323, i64 0
  %361 = insertelement <2 x bfloat> %360, bfloat %324, i64 1
  %362 = bitcast <2 x bfloat> %361 to i32
  %363 = insertelement <2 x bfloat> poison, bfloat %325, i64 0
  %364 = insertelement <2 x bfloat> %363, bfloat %326, i64 1
  %365 = bitcast <2 x bfloat> %364 to i32
  %366 = insertelement <2 x bfloat> poison, bfloat %327, i64 0
  %367 = insertelement <2 x bfloat> %366, bfloat %328, i64 1
  %368 = bitcast <2 x bfloat> %367 to i32
  %369 = insertelement <2 x bfloat> poison, bfloat %329, i64 0
  %370 = insertelement <2 x bfloat> %369, bfloat %330, i64 1
  %371 = bitcast <2 x bfloat> %370 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %362, i32 %365, i32 %368, i32 %371, ptr addrspace(1) %356) #7
  %372 = insertelement <2 x bfloat> poison, bfloat %331, i64 0
  %373 = insertelement <2 x bfloat> %372, bfloat %332, i64 1
  %374 = bitcast <2 x bfloat> %373 to i32
  %375 = insertelement <2 x bfloat> poison, bfloat %333, i64 0
  %376 = insertelement <2 x bfloat> %375, bfloat %334, i64 1
  %377 = bitcast <2 x bfloat> %376 to i32
  %378 = insertelement <2 x bfloat> poison, bfloat %335, i64 0
  %379 = insertelement <2 x bfloat> %378, bfloat %336, i64 1
  %380 = bitcast <2 x bfloat> %379 to i32
  %381 = insertelement <2 x bfloat> poison, bfloat %337, i64 0
  %382 = insertelement <2 x bfloat> %381, bfloat %338, i64 1
  %383 = bitcast <2 x bfloat> %382 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %374, i32 %377, i32 %380, i32 %383, ptr addrspace(1) %357) #7
  %384 = insertelement <2 x bfloat> poison, bfloat %339, i64 0
  %385 = insertelement <2 x bfloat> %384, bfloat %340, i64 1
  %386 = bitcast <2 x bfloat> %385 to i32
  %387 = insertelement <2 x bfloat> poison, bfloat %341, i64 0
  %388 = insertelement <2 x bfloat> %387, bfloat %342, i64 1
  %389 = bitcast <2 x bfloat> %388 to i32
  %390 = insertelement <2 x bfloat> poison, bfloat %343, i64 0
  %391 = insertelement <2 x bfloat> %390, bfloat %344, i64 1
  %392 = bitcast <2 x bfloat> %391 to i32
  %393 = insertelement <2 x bfloat> poison, bfloat %345, i64 0
  %394 = insertelement <2 x bfloat> %393, bfloat %346, i64 1
  %395 = bitcast <2 x bfloat> %394 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %386, i32 %389, i32 %392, i32 %395, ptr addrspace(1) %358) #7
  %396 = insertelement <2 x bfloat> poison, bfloat %347, i64 0
  %397 = insertelement <2 x bfloat> %396, bfloat %348, i64 1
  %398 = bitcast <2 x bfloat> %397 to i32
  %399 = insertelement <2 x bfloat> poison, bfloat %349, i64 0
  %400 = insertelement <2 x bfloat> %399, bfloat %350, i64 1
  %401 = bitcast <2 x bfloat> %400 to i32
  %402 = insertelement <2 x bfloat> poison, bfloat %351, i64 0
  %403 = insertelement <2 x bfloat> %402, bfloat %352, i64 1
  %404 = bitcast <2 x bfloat> %403 to i32
  %405 = insertelement <2 x bfloat> poison, bfloat %353, i64 0
  %406 = insertelement <2 x bfloat> %405, bfloat %354, i64 1
  %407 = bitcast <2 x bfloat> %406 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %398, i32 %401, i32 %404, i32 %407, ptr addrspace(1) %359) #7
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare bfloat @llvm.maximum.bf16(bfloat, bfloat) #9

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_exponential(ptr noalias align 256 captures(none) dereferenceable(1073741824) %0, ptr noalias readnone align 256 captures(none) dereferenceable(1073741824) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !9
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %6 = shl nuw nsw i32 %5, 2
  %7 = shl nuw nsw i32 %4, 9
  %8 = or disjoint i32 %6, %7
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %9
  %11 = load <4 x bfloat>, ptr addrspace(1) %10, align 8
  %12 = extractelement <4 x bfloat> %11, i32 0
  %13 = extractelement <4 x bfloat> %11, i32 1
  %14 = extractelement <4 x bfloat> %11, i32 2
  %15 = extractelement <4 x bfloat> %11, i32 3
  %16 = fpext bfloat %12 to float
  %17 = fmul float %16, 0x3FF7154760000000
  %18 = tail call float @llvm.nvvm.ex2.approx.f32(float %17)
  %19 = fptrunc float %18 to bfloat
  %20 = fpext bfloat %13 to float
  %21 = fmul float %20, 0x3FF7154760000000
  %22 = tail call float @llvm.nvvm.ex2.approx.f32(float %21)
  %23 = fptrunc float %22 to bfloat
  %24 = fpext bfloat %14 to float
  %25 = fmul float %24, 0x3FF7154760000000
  %26 = tail call float @llvm.nvvm.ex2.approx.f32(float %25)
  %27 = fptrunc float %26 to bfloat
  %28 = fpext bfloat %15 to float
  %29 = fmul float %28, 0x3FF7154760000000
  %30 = tail call float @llvm.nvvm.ex2.approx.f32(float %29)
  %31 = fptrunc float %30 to bfloat
  %32 = insertelement <4 x bfloat> poison, bfloat %19, i64 0
  %33 = insertelement <4 x bfloat> %32, bfloat %23, i64 1
  %34 = insertelement <4 x bfloat> %33, bfloat %27, i64 2
  %35 = insertelement <4 x bfloat> %34, bfloat %31, i64 3
  store <4 x bfloat> %35, ptr addrspace(1) %10, align 8
  ret void
}

; Function Attrs: nounwind
define ptx_kernel void @triton_softmax_3(ptr noalias align 256 dereferenceable(1073741824) %arg0, ptr noalias align 256 dereferenceable(1073741824) %arg1) local_unnamed_addr #4 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = zext i32 %3 to i64
  %5 = tail call range(i32 0, 128) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = shl i32 %5, 3
  %7 = zext i32 %6 to i64
  %8 = shl i64 %4, 12
  %9 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %8
  %10 = getelementptr [2 x i8], ptr addrspace(1) %9, i64 %7
  %11 = getelementptr i8, ptr addrspace(1) %10, i64 2048
  %12 = getelementptr i8, ptr addrspace(1) %10, i64 4096
  %13 = getelementptr i8, ptr addrspace(1) %10, i64 6144
  %14 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %10) #7
  %15 = extractvalue { i32, i32, i32, i32 } %14, 0
  %16 = bitcast i32 %15 to <2 x bfloat>
  %17 = extractvalue { i32, i32, i32, i32 } %14, 1
  %18 = bitcast i32 %17 to <2 x bfloat>
  %19 = extractvalue { i32, i32, i32, i32 } %14, 2
  %20 = bitcast i32 %19 to <2 x bfloat>
  %21 = extractvalue { i32, i32, i32, i32 } %14, 3
  %22 = bitcast i32 %21 to <2 x bfloat>
  %23 = extractelement <2 x bfloat> %16, i64 0
  %24 = extractelement <2 x bfloat> %16, i64 1
  %25 = extractelement <2 x bfloat> %18, i64 0
  %26 = extractelement <2 x bfloat> %18, i64 1
  %27 = extractelement <2 x bfloat> %20, i64 0
  %28 = extractelement <2 x bfloat> %20, i64 1
  %29 = extractelement <2 x bfloat> %22, i64 0
  %30 = extractelement <2 x bfloat> %22, i64 1
  %31 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %11) #7
  %32 = extractvalue { i32, i32, i32, i32 } %31, 0
  %33 = bitcast i32 %32 to <2 x bfloat>
  %34 = extractvalue { i32, i32, i32, i32 } %31, 1
  %35 = bitcast i32 %34 to <2 x bfloat>
  %36 = extractvalue { i32, i32, i32, i32 } %31, 2
  %37 = bitcast i32 %36 to <2 x bfloat>
  %38 = extractvalue { i32, i32, i32, i32 } %31, 3
  %39 = bitcast i32 %38 to <2 x bfloat>
  %40 = extractelement <2 x bfloat> %33, i64 0
  %41 = extractelement <2 x bfloat> %33, i64 1
  %42 = extractelement <2 x bfloat> %35, i64 0
  %43 = extractelement <2 x bfloat> %35, i64 1
  %44 = extractelement <2 x bfloat> %37, i64 0
  %45 = extractelement <2 x bfloat> %37, i64 1
  %46 = extractelement <2 x bfloat> %39, i64 0
  %47 = extractelement <2 x bfloat> %39, i64 1
  %48 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %12) #7
  %49 = extractvalue { i32, i32, i32, i32 } %48, 0
  %50 = bitcast i32 %49 to <2 x bfloat>
  %51 = extractvalue { i32, i32, i32, i32 } %48, 1
  %52 = bitcast i32 %51 to <2 x bfloat>
  %53 = extractvalue { i32, i32, i32, i32 } %48, 2
  %54 = bitcast i32 %53 to <2 x bfloat>
  %55 = extractvalue { i32, i32, i32, i32 } %48, 3
  %56 = bitcast i32 %55 to <2 x bfloat>
  %57 = extractelement <2 x bfloat> %50, i64 0
  %58 = extractelement <2 x bfloat> %50, i64 1
  %59 = extractelement <2 x bfloat> %52, i64 0
  %60 = extractelement <2 x bfloat> %52, i64 1
  %61 = extractelement <2 x bfloat> %54, i64 0
  %62 = extractelement <2 x bfloat> %54, i64 1
  %63 = extractelement <2 x bfloat> %56, i64 0
  %64 = extractelement <2 x bfloat> %56, i64 1
  %65 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0; mov.u32 $1, 0x0; mov.u32 $2, 0x0; mov.u32 $3, 0x0; ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %13) #7
  %66 = extractvalue { i32, i32, i32, i32 } %65, 0
  %67 = bitcast i32 %66 to <2 x bfloat>
  %68 = extractvalue { i32, i32, i32, i32 } %65, 1
  %69 = bitcast i32 %68 to <2 x bfloat>
  %70 = extractvalue { i32, i32, i32, i32 } %65, 2
  %71 = bitcast i32 %70 to <2 x bfloat>
  %72 = extractvalue { i32, i32, i32, i32 } %65, 3
  %73 = bitcast i32 %72 to <2 x bfloat>
  %74 = extractelement <2 x bfloat> %67, i64 0
  %75 = extractelement <2 x bfloat> %67, i64 1
  %76 = extractelement <2 x bfloat> %69, i64 0
  %77 = extractelement <2 x bfloat> %69, i64 1
  %78 = extractelement <2 x bfloat> %71, i64 0
  %79 = extractelement <2 x bfloat> %71, i64 1
  %80 = extractelement <2 x bfloat> %73, i64 0
  %81 = extractelement <2 x bfloat> %73, i64 1
  %82 = fpext bfloat %23 to float
  %83 = fpext bfloat %24 to float
  %84 = fpext bfloat %25 to float
  %85 = fpext bfloat %26 to float
  %86 = fpext bfloat %27 to float
  %87 = fpext bfloat %28 to float
  %88 = fpext bfloat %29 to float
  %89 = fpext bfloat %30 to float
  %90 = fpext bfloat %40 to float
  %91 = fpext bfloat %41 to float
  %92 = fpext bfloat %42 to float
  %93 = fpext bfloat %43 to float
  %94 = fpext bfloat %44 to float
  %95 = fpext bfloat %45 to float
  %96 = fpext bfloat %46 to float
  %97 = fpext bfloat %47 to float
  %98 = fpext bfloat %57 to float
  %99 = fpext bfloat %58 to float
  %100 = fpext bfloat %59 to float
  %101 = fpext bfloat %60 to float
  %102 = fpext bfloat %61 to float
  %103 = fpext bfloat %62 to float
  %104 = fpext bfloat %63 to float
  %105 = fpext bfloat %64 to float
  %106 = fpext bfloat %74 to float
  %107 = fpext bfloat %75 to float
  %108 = fpext bfloat %76 to float
  %109 = fpext bfloat %77 to float
  %110 = fpext bfloat %78 to float
  %111 = fpext bfloat %79 to float
  %112 = fpext bfloat %80 to float
  %113 = fpext bfloat %81 to float
  %114 = shl nuw nsw i32 %5, 4
  %115 = zext nneg i32 %114 to i64
  %116 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %115
  %117 = insertelement <4 x float> poison, float %82, i64 0
  %118 = insertelement <4 x float> %117, float %83, i64 1
  %119 = insertelement <4 x float> %118, float %84, i64 2
  %120 = insertelement <4 x float> %119, float %85, i64 3
  store <4 x float> %120, ptr addrspace(3) %116, align 16
  %121 = xor i32 %114, 2112
  %122 = zext nneg i32 %121 to i64
  %123 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %122
  %124 = insertelement <4 x float> poison, float %86, i64 0
  %125 = insertelement <4 x float> %124, float %87, i64 1
  %126 = insertelement <4 x float> %125, float %88, i64 2
  %127 = insertelement <4 x float> %126, float %89, i64 3
  store <4 x float> %127, ptr addrspace(3) %123, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %128 = shl nuw nsw i32 %5, 5
  %129 = and i32 %128, 768
  %130 = and i32 %6, 48
  %131 = and i32 %5, 96
  %132 = shl nuw nsw i32 %131, 1
  %133 = and i32 %5, 1
  %134 = icmp eq i32 %133, 0
  %135 = select i1 %134, i32 0, i32 2112
  %136 = or disjoint i32 %129, %130
  %137 = xor i32 %135, %132
  %138 = or disjoint i32 %136, %137
  %139 = zext nneg i32 %138 to i64
  %140 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %139
  %141 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %140)
  %142 = extractvalue { i32, i32, i32, i32 } %141, 0
  %143 = bitcast i32 %142 to float
  %144 = extractvalue { i32, i32, i32, i32 } %141, 1
  %145 = bitcast i32 %144 to float
  %146 = extractvalue { i32, i32, i32, i32 } %141, 2
  %147 = bitcast i32 %146 to float
  %148 = extractvalue { i32, i32, i32, i32 } %141, 3
  %149 = bitcast i32 %148 to float
  %150 = getelementptr inbounds nuw i8, ptr addrspace(3) %140, i64 1024
  %151 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %150)
  %152 = extractvalue { i32, i32, i32, i32 } %151, 0
  %153 = bitcast i32 %152 to float
  %154 = extractvalue { i32, i32, i32, i32 } %151, 1
  %155 = bitcast i32 %154 to float
  %156 = extractvalue { i32, i32, i32, i32 } %151, 2
  %157 = bitcast i32 %156 to float
  %158 = extractvalue { i32, i32, i32, i32 } %151, 3
  %159 = bitcast i32 %158 to float
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %160 = insertelement <4 x float> poison, float %90, i64 0
  %161 = insertelement <4 x float> %160, float %91, i64 1
  %162 = insertelement <4 x float> %161, float %92, i64 2
  %163 = insertelement <4 x float> %162, float %93, i64 3
  store <4 x float> %163, ptr addrspace(3) %116, align 16
  %164 = insertelement <4 x float> poison, float %94, i64 0
  %165 = insertelement <4 x float> %164, float %95, i64 1
  %166 = insertelement <4 x float> %165, float %96, i64 2
  %167 = insertelement <4 x float> %166, float %97, i64 3
  store <4 x float> %167, ptr addrspace(3) %123, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %168 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %140)
  %169 = extractvalue { i32, i32, i32, i32 } %168, 0
  %170 = bitcast i32 %169 to float
  %171 = extractvalue { i32, i32, i32, i32 } %168, 1
  %172 = bitcast i32 %171 to float
  %173 = extractvalue { i32, i32, i32, i32 } %168, 2
  %174 = bitcast i32 %173 to float
  %175 = extractvalue { i32, i32, i32, i32 } %168, 3
  %176 = bitcast i32 %175 to float
  %177 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %150)
  %178 = extractvalue { i32, i32, i32, i32 } %177, 0
  %179 = bitcast i32 %178 to float
  %180 = extractvalue { i32, i32, i32, i32 } %177, 1
  %181 = bitcast i32 %180 to float
  %182 = extractvalue { i32, i32, i32, i32 } %177, 2
  %183 = bitcast i32 %182 to float
  %184 = extractvalue { i32, i32, i32, i32 } %177, 3
  %185 = bitcast i32 %184 to float
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %186 = insertelement <4 x float> poison, float %98, i64 0
  %187 = insertelement <4 x float> %186, float %99, i64 1
  %188 = insertelement <4 x float> %187, float %100, i64 2
  %189 = insertelement <4 x float> %188, float %101, i64 3
  store <4 x float> %189, ptr addrspace(3) %116, align 16
  %190 = insertelement <4 x float> poison, float %102, i64 0
  %191 = insertelement <4 x float> %190, float %103, i64 1
  %192 = insertelement <4 x float> %191, float %104, i64 2
  %193 = insertelement <4 x float> %192, float %105, i64 3
  store <4 x float> %193, ptr addrspace(3) %123, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %194 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %140)
  %195 = extractvalue { i32, i32, i32, i32 } %194, 0
  %196 = bitcast i32 %195 to float
  %197 = extractvalue { i32, i32, i32, i32 } %194, 1
  %198 = bitcast i32 %197 to float
  %199 = extractvalue { i32, i32, i32, i32 } %194, 2
  %200 = bitcast i32 %199 to float
  %201 = extractvalue { i32, i32, i32, i32 } %194, 3
  %202 = bitcast i32 %201 to float
  %203 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %150)
  %204 = extractvalue { i32, i32, i32, i32 } %203, 0
  %205 = bitcast i32 %204 to float
  %206 = extractvalue { i32, i32, i32, i32 } %203, 1
  %207 = bitcast i32 %206 to float
  %208 = extractvalue { i32, i32, i32, i32 } %203, 2
  %209 = bitcast i32 %208 to float
  %210 = extractvalue { i32, i32, i32, i32 } %203, 3
  %211 = bitcast i32 %210 to float
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %212 = insertelement <4 x float> poison, float %106, i64 0
  %213 = insertelement <4 x float> %212, float %107, i64 1
  %214 = insertelement <4 x float> %213, float %108, i64 2
  %215 = insertelement <4 x float> %214, float %109, i64 3
  store <4 x float> %215, ptr addrspace(3) %116, align 16
  %216 = insertelement <4 x float> poison, float %110, i64 0
  %217 = insertelement <4 x float> %216, float %111, i64 1
  %218 = insertelement <4 x float> %217, float %112, i64 2
  %219 = insertelement <4 x float> %218, float %113, i64 3
  store <4 x float> %219, ptr addrspace(3) %123, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %220 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %140)
  %221 = extractvalue { i32, i32, i32, i32 } %220, 0
  %222 = bitcast i32 %221 to float
  %223 = extractvalue { i32, i32, i32, i32 } %220, 1
  %224 = bitcast i32 %223 to float
  %225 = extractvalue { i32, i32, i32, i32 } %220, 2
  %226 = bitcast i32 %225 to float
  %227 = extractvalue { i32, i32, i32, i32 } %220, 3
  %228 = bitcast i32 %227 to float
  %229 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %150)
  %230 = extractvalue { i32, i32, i32, i32 } %229, 0
  %231 = bitcast i32 %230 to float
  %232 = extractvalue { i32, i32, i32, i32 } %229, 1
  %233 = bitcast i32 %232 to float
  %234 = extractvalue { i32, i32, i32, i32 } %229, 2
  %235 = bitcast i32 %234 to float
  %236 = extractvalue { i32, i32, i32, i32 } %229, 3
  %237 = bitcast i32 %236 to float
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %238 = fadd float %143, %145
  %239 = fadd float %147, %149
  %240 = fadd float %153, %155
  %241 = fadd float %157, %159
  %242 = fadd float %170, %172
  %243 = fadd float %174, %176
  %244 = fadd float %179, %181
  %245 = fadd float %183, %185
  %246 = fadd float %196, %198
  %247 = fadd float %200, %202
  %248 = fadd float %205, %207
  %249 = fadd float %209, %211
  %250 = fadd float %222, %224
  %251 = fadd float %226, %228
  %252 = fadd float %231, %233
  %253 = fadd float %235, %237
  %254 = fadd float %238, %239
  %255 = fadd float %240, %241
  %256 = fadd float %242, %243
  %257 = fadd float %244, %245
  %258 = fadd float %246, %247
  %259 = fadd float %248, %249
  %260 = fadd float %250, %251
  %261 = fadd float %252, %253
  %262 = fadd float %254, %255
  %263 = fadd float %256, %257
  %264 = fadd float %258, %259
  %265 = fadd float %260, %261
  %266 = fadd float %262, %263
  %267 = fadd float %264, %265
  %268 = fadd float %266, %267
  %269 = bitcast float %268 to i32
  %270 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %269, i32 16, i32 31)
  %271 = bitcast i32 %270 to float
  %272 = fadd float %268, %271
  %273 = bitcast float %272 to i32
  %274 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %273, i32 8, i32 31)
  %275 = bitcast i32 %274 to float
  %276 = fadd float %272, %275
  %277 = bitcast float %276 to i32
  %278 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %277, i32 4, i32 31)
  %279 = bitcast i32 %278 to float
  %280 = fadd float %276, %279
  %281 = bitcast float %280 to i32
  %282 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %281, i32 2, i32 31)
  %283 = bitcast i32 %282 to float
  %284 = fadd float %280, %283
  %285 = bitcast float %284 to i32
  %286 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %285, i32 1, i32 31)
  %287 = bitcast i32 %286 to float
  %288 = fadd float %284, %287
  %289 = lshr exact i32 %131, 3
  %290 = zext nneg i32 %289 to i64
  %291 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %290
  store float %288, ptr addrspace(3) %291, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %292 = shl nuw nsw i32 %5, 2
  %293 = and i32 %292, 12
  %294 = zext nneg i32 %293 to i64
  %295 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %294
  %296 = load i32, ptr addrspace(3) %295, align 4
  %297 = bitcast i32 %296 to float
  %298 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %296, i32 2, i32 31)
  %299 = bitcast i32 %298 to float
  %300 = fadd float %297, %299
  %301 = bitcast float %300 to i32
  %302 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %301, i32 1, i32 31)
  %303 = bitcast i32 %302 to float
  %304 = fadd float %300, %303
  %305 = fptrunc float %304 to bfloat
  %306 = fpext bfloat %305 to float
  %307 = tail call float @llvm.nvvm.div.full(float %82, float %306)
  %308 = tail call float @llvm.nvvm.div.full(float %83, float %306)
  %309 = tail call float @llvm.nvvm.div.full(float %84, float %306)
  %310 = tail call float @llvm.nvvm.div.full(float %85, float %306)
  %311 = tail call float @llvm.nvvm.div.full(float %86, float %306)
  %312 = tail call float @llvm.nvvm.div.full(float %87, float %306)
  %313 = tail call float @llvm.nvvm.div.full(float %88, float %306)
  %314 = tail call float @llvm.nvvm.div.full(float %89, float %306)
  %315 = tail call float @llvm.nvvm.div.full(float %90, float %306)
  %316 = tail call float @llvm.nvvm.div.full(float %91, float %306)
  %317 = tail call float @llvm.nvvm.div.full(float %92, float %306)
  %318 = tail call float @llvm.nvvm.div.full(float %93, float %306)
  %319 = tail call float @llvm.nvvm.div.full(float %94, float %306)
  %320 = tail call float @llvm.nvvm.div.full(float %95, float %306)
  %321 = tail call float @llvm.nvvm.div.full(float %96, float %306)
  %322 = tail call float @llvm.nvvm.div.full(float %97, float %306)
  %323 = tail call float @llvm.nvvm.div.full(float %98, float %306)
  %324 = tail call float @llvm.nvvm.div.full(float %99, float %306)
  %325 = tail call float @llvm.nvvm.div.full(float %100, float %306)
  %326 = tail call float @llvm.nvvm.div.full(float %101, float %306)
  %327 = tail call float @llvm.nvvm.div.full(float %102, float %306)
  %328 = tail call float @llvm.nvvm.div.full(float %103, float %306)
  %329 = tail call float @llvm.nvvm.div.full(float %104, float %306)
  %330 = tail call float @llvm.nvvm.div.full(float %105, float %306)
  %331 = tail call float @llvm.nvvm.div.full(float %106, float %306)
  %332 = tail call float @llvm.nvvm.div.full(float %107, float %306)
  %333 = tail call float @llvm.nvvm.div.full(float %108, float %306)
  %334 = tail call float @llvm.nvvm.div.full(float %109, float %306)
  %335 = tail call float @llvm.nvvm.div.full(float %110, float %306)
  %336 = tail call float @llvm.nvvm.div.full(float %111, float %306)
  %337 = tail call float @llvm.nvvm.div.full(float %112, float %306)
  %338 = tail call float @llvm.nvvm.div.full(float %113, float %306)
  %339 = fptrunc float %307 to bfloat
  %340 = fptrunc float %308 to bfloat
  %341 = fptrunc float %309 to bfloat
  %342 = fptrunc float %310 to bfloat
  %343 = fptrunc float %311 to bfloat
  %344 = fptrunc float %312 to bfloat
  %345 = fptrunc float %313 to bfloat
  %346 = fptrunc float %314 to bfloat
  %347 = fptrunc float %315 to bfloat
  %348 = fptrunc float %316 to bfloat
  %349 = fptrunc float %317 to bfloat
  %350 = fptrunc float %318 to bfloat
  %351 = fptrunc float %319 to bfloat
  %352 = fptrunc float %320 to bfloat
  %353 = fptrunc float %321 to bfloat
  %354 = fptrunc float %322 to bfloat
  %355 = fptrunc float %323 to bfloat
  %356 = fptrunc float %324 to bfloat
  %357 = fptrunc float %325 to bfloat
  %358 = fptrunc float %326 to bfloat
  %359 = fptrunc float %327 to bfloat
  %360 = fptrunc float %328 to bfloat
  %361 = fptrunc float %329 to bfloat
  %362 = fptrunc float %330 to bfloat
  %363 = fptrunc float %331 to bfloat
  %364 = fptrunc float %332 to bfloat
  %365 = fptrunc float %333 to bfloat
  %366 = fptrunc float %334 to bfloat
  %367 = fptrunc float %335 to bfloat
  %368 = fptrunc float %336 to bfloat
  %369 = fptrunc float %337 to bfloat
  %370 = fptrunc float %338 to bfloat
  %371 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %8
  %372 = getelementptr [2 x i8], ptr addrspace(1) %371, i64 %7
  %373 = getelementptr i8, ptr addrspace(1) %372, i64 2048
  %374 = getelementptr i8, ptr addrspace(1) %372, i64 4096
  %375 = getelementptr i8, ptr addrspace(1) %372, i64 6144
  %376 = insertelement <2 x bfloat> poison, bfloat %339, i64 0
  %377 = insertelement <2 x bfloat> %376, bfloat %340, i64 1
  %378 = bitcast <2 x bfloat> %377 to i32
  %379 = insertelement <2 x bfloat> poison, bfloat %341, i64 0
  %380 = insertelement <2 x bfloat> %379, bfloat %342, i64 1
  %381 = bitcast <2 x bfloat> %380 to i32
  %382 = insertelement <2 x bfloat> poison, bfloat %343, i64 0
  %383 = insertelement <2 x bfloat> %382, bfloat %344, i64 1
  %384 = bitcast <2 x bfloat> %383 to i32
  %385 = insertelement <2 x bfloat> poison, bfloat %345, i64 0
  %386 = insertelement <2 x bfloat> %385, bfloat %346, i64 1
  %387 = bitcast <2 x bfloat> %386 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %378, i32 %381, i32 %384, i32 %387, ptr addrspace(1) %372) #7
  %388 = insertelement <2 x bfloat> poison, bfloat %347, i64 0
  %389 = insertelement <2 x bfloat> %388, bfloat %348, i64 1
  %390 = bitcast <2 x bfloat> %389 to i32
  %391 = insertelement <2 x bfloat> poison, bfloat %349, i64 0
  %392 = insertelement <2 x bfloat> %391, bfloat %350, i64 1
  %393 = bitcast <2 x bfloat> %392 to i32
  %394 = insertelement <2 x bfloat> poison, bfloat %351, i64 0
  %395 = insertelement <2 x bfloat> %394, bfloat %352, i64 1
  %396 = bitcast <2 x bfloat> %395 to i32
  %397 = insertelement <2 x bfloat> poison, bfloat %353, i64 0
  %398 = insertelement <2 x bfloat> %397, bfloat %354, i64 1
  %399 = bitcast <2 x bfloat> %398 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %390, i32 %393, i32 %396, i32 %399, ptr addrspace(1) %373) #7
  %400 = insertelement <2 x bfloat> poison, bfloat %355, i64 0
  %401 = insertelement <2 x bfloat> %400, bfloat %356, i64 1
  %402 = bitcast <2 x bfloat> %401 to i32
  %403 = insertelement <2 x bfloat> poison, bfloat %357, i64 0
  %404 = insertelement <2 x bfloat> %403, bfloat %358, i64 1
  %405 = bitcast <2 x bfloat> %404 to i32
  %406 = insertelement <2 x bfloat> poison, bfloat %359, i64 0
  %407 = insertelement <2 x bfloat> %406, bfloat %360, i64 1
  %408 = bitcast <2 x bfloat> %407 to i32
  %409 = insertelement <2 x bfloat> poison, bfloat %361, i64 0
  %410 = insertelement <2 x bfloat> %409, bfloat %362, i64 1
  %411 = bitcast <2 x bfloat> %410 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %402, i32 %405, i32 %408, i32 %411, ptr addrspace(1) %374) #7
  %412 = insertelement <2 x bfloat> poison, bfloat %363, i64 0
  %413 = insertelement <2 x bfloat> %412, bfloat %364, i64 1
  %414 = bitcast <2 x bfloat> %413 to i32
  %415 = insertelement <2 x bfloat> poison, bfloat %365, i64 0
  %416 = insertelement <2 x bfloat> %415, bfloat %366, i64 1
  %417 = bitcast <2 x bfloat> %416 to i32
  %418 = insertelement <2 x bfloat> poison, bfloat %367, i64 0
  %419 = insertelement <2 x bfloat> %418, bfloat %368, i64 1
  %420 = bitcast <2 x bfloat> %419 to i32
  %421 = insertelement <2 x bfloat> poison, bfloat %369, i64 0
  %422 = insertelement <2 x bfloat> %421, bfloat %370, i64 1
  %423 = bitcast <2 x bfloat> %422 to i32
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %414, i32 %417, i32 %420, i32 %423, ptr addrspace(1) %375) #7
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.div.full(float, float) #10

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8388608) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8388608) %1) local_unnamed_addr #11 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !10
  %7 = and i32 %5, 31
  %8 = lshr i32 %5, 6
  %9 = add nuw nsw i32 %7, %8
  %10 = shl nuw nsw i32 %9, 1
  %11 = and i32 %10, 62
  %12 = lshr i32 %5, 5
  %13 = shl nuw nsw i32 %12, 6
  %14 = or disjoint i32 %11, %13
  %15 = lshr i32 %6, 1
  %16 = shl nuw nsw i32 %15, 6
  %17 = and i32 %16, 4032
  %18 = shl nuw nsw i32 %7, 1
  %19 = and i32 %6, 1
  %20 = shl nuw nsw i32 %19, 18
  %21 = or disjoint i32 %17, %20
  %22 = shl nuw nsw i32 %12, 12
  %23 = or disjoint i32 %21, %22
  %24 = shl nuw nsw i32 %6, 12
  %25 = and i32 %24, 3670016
  %26 = or disjoint i32 %23, %25
  %27 = or disjoint i32 %26, %18
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %28
  %30 = load <2 x bfloat>, ptr addrspace(1) %29, align 2, !invariant.load !6
  %31 = zext nneg i32 %14 to i64
  %32 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %31
  store <2 x bfloat> %30, ptr addrspace(3) %32, align 2
  %33 = add nuw nsw i32 %10, 4
  %34 = and i32 %33, 62
  %35 = or disjoint i32 %34, %13
  %36 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 32768
  %37 = load <2 x bfloat>, ptr addrspace(1) %36, align 2, !invariant.load !6
  %38 = zext nneg i32 %35 to i64
  %39 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %38
  %40 = getelementptr inbounds i8, ptr addrspace(3) %39, i64 512
  store <2 x bfloat> %37, ptr addrspace(3) %40, align 2
  %41 = add i32 %33, 4
  %42 = and i32 %41, 62
  %43 = or disjoint i32 %42, %13
  %44 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 65536
  %45 = load <2 x bfloat>, ptr addrspace(1) %44, align 2, !invariant.load !6
  %46 = zext nneg i32 %43 to i64
  %47 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %46
  %48 = getelementptr inbounds i8, ptr addrspace(3) %47, i64 1024
  store <2 x bfloat> %45, ptr addrspace(3) %48, align 2
  %49 = add i32 %33, 8
  %50 = and i32 %49, 62
  %51 = or disjoint i32 %50, %13
  %52 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 98304
  %53 = load <2 x bfloat>, ptr addrspace(1) %52, align 2, !invariant.load !6
  %54 = zext nneg i32 %51 to i64
  %55 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %54
  %56 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 1536
  store <2 x bfloat> %53, ptr addrspace(3) %56, align 2
  %57 = add i32 %33, 12
  %58 = and i32 %57, 62
  %59 = or disjoint i32 %58, %13
  %60 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 131072
  %61 = load <2 x bfloat>, ptr addrspace(1) %60, align 2, !invariant.load !6
  %62 = zext nneg i32 %59 to i64
  %63 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %62
  %64 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 2048
  store <2 x bfloat> %61, ptr addrspace(3) %64, align 2
  %65 = add i32 %33, 16
  %66 = and i32 %65, 62
  %67 = or disjoint i32 %66, %13
  %68 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 163840
  %69 = load <2 x bfloat>, ptr addrspace(1) %68, align 2, !invariant.load !6
  %70 = zext nneg i32 %67 to i64
  %71 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %70
  %72 = getelementptr inbounds i8, ptr addrspace(3) %71, i64 2560
  store <2 x bfloat> %69, ptr addrspace(3) %72, align 2
  %73 = add i32 %33, 20
  %74 = and i32 %73, 62
  %75 = or disjoint i32 %74, %13
  %76 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 196608
  %77 = load <2 x bfloat>, ptr addrspace(1) %76, align 2, !invariant.load !6
  %78 = zext nneg i32 %75 to i64
  %79 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %78
  %80 = getelementptr inbounds i8, ptr addrspace(3) %79, i64 3072
  store <2 x bfloat> %77, ptr addrspace(3) %80, align 2
  %81 = add i32 %33, 24
  %82 = and i32 %81, 62
  %83 = or disjoint i32 %82, %13
  %84 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 229376
  %85 = load <2 x bfloat>, ptr addrspace(1) %84, align 2, !invariant.load !6
  %86 = zext nneg i32 %83 to i64
  %87 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %86
  %88 = getelementptr inbounds i8, ptr addrspace(3) %87, i64 3584
  store <2 x bfloat> %85, ptr addrspace(3) %88, align 2
  %89 = xor i32 %14, 2080
  %90 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 262144
  %91 = load <2 x bfloat>, ptr addrspace(1) %90, align 2, !invariant.load !6
  %92 = zext nneg i32 %89 to i64
  %93 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %92
  store <2 x bfloat> %91, ptr addrspace(3) %93, align 2
  %94 = add i32 %33, 32
  %95 = and i32 %94, 62
  %96 = or disjoint i32 %95, %13
  %97 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 294912
  %98 = load <2 x bfloat>, ptr addrspace(1) %97, align 2, !invariant.load !6
  %99 = zext nneg i32 %96 to i64
  %100 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %99
  %101 = getelementptr inbounds i8, ptr addrspace(3) %100, i64 4608
  store <2 x bfloat> %98, ptr addrspace(3) %101, align 2
  %102 = add i32 %33, 36
  %103 = and i32 %102, 62
  %104 = or disjoint i32 %103, %13
  %105 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 327680
  %106 = load <2 x bfloat>, ptr addrspace(1) %105, align 2, !invariant.load !6
  %107 = zext nneg i32 %104 to i64
  %108 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %107
  %109 = getelementptr inbounds i8, ptr addrspace(3) %108, i64 5120
  store <2 x bfloat> %106, ptr addrspace(3) %109, align 2
  %110 = add i32 %33, 40
  %111 = and i32 %110, 62
  %112 = or disjoint i32 %111, %13
  %113 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 360448
  %114 = load <2 x bfloat>, ptr addrspace(1) %113, align 2, !invariant.load !6
  %115 = zext nneg i32 %112 to i64
  %116 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %115
  %117 = getelementptr inbounds i8, ptr addrspace(3) %116, i64 5632
  store <2 x bfloat> %114, ptr addrspace(3) %117, align 2
  %118 = add i32 %33, 44
  %119 = and i32 %118, 62
  %120 = or disjoint i32 %119, %13
  %121 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 393216
  %122 = load <2 x bfloat>, ptr addrspace(1) %121, align 2, !invariant.load !6
  %123 = zext nneg i32 %120 to i64
  %124 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %123
  %125 = getelementptr inbounds i8, ptr addrspace(3) %124, i64 6144
  store <2 x bfloat> %122, ptr addrspace(3) %125, align 2
  %126 = add i32 %33, 48
  %127 = and i32 %126, 62
  %128 = or disjoint i32 %127, %13
  %129 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 425984
  %130 = load <2 x bfloat>, ptr addrspace(1) %129, align 2, !invariant.load !6
  %131 = zext nneg i32 %128 to i64
  %132 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %131
  %133 = getelementptr inbounds i8, ptr addrspace(3) %132, i64 6656
  store <2 x bfloat> %130, ptr addrspace(3) %133, align 2
  %134 = add i32 %33, 52
  %135 = and i32 %134, 62
  %136 = or disjoint i32 %135, %13
  %137 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 458752
  %138 = load <2 x bfloat>, ptr addrspace(1) %137, align 2, !invariant.load !6
  %139 = zext nneg i32 %136 to i64
  %140 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %139
  %141 = getelementptr inbounds i8, ptr addrspace(3) %140, i64 7168
  store <2 x bfloat> %138, ptr addrspace(3) %141, align 2
  %142 = add i32 %33, 56
  %143 = and i32 %142, 62
  %144 = or disjoint i32 %143, %13
  %145 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 491520
  %146 = load <2 x bfloat>, ptr addrspace(1) %145, align 2, !invariant.load !6
  %147 = zext nneg i32 %144 to i64
  %148 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %147
  %149 = getelementptr inbounds i8, ptr addrspace(3) %148, i64 7680
  store <2 x bfloat> %146, ptr addrspace(3) %149, align 2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %150 = shl nuw nsw i32 %7, 7
  %151 = add nuw nsw i32 %7, %12
  %152 = or disjoint i32 %150, 64
  %153 = shl nuw nsw i32 %19, 6
  %154 = shl nuw nsw i32 %15, 13
  %155 = or disjoint i32 %153, %154
  %156 = shl nuw nsw i32 %12, 8
  %157 = or disjoint i32 %155, %156
  %158 = or disjoint i32 %157, %18
  %159 = shl nuw nsw i32 %151, 1
  %160 = and i32 %159, 62
  %161 = or disjoint i32 %160, %150
  %162 = zext nneg i32 %161 to i64
  %163 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %162
  %164 = load <2 x bfloat>, ptr addrspace(3) %163, align 2
  %165 = or i32 %159, %152
  %166 = zext nneg i32 %165 to i64
  %167 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %166
  %168 = load <2 x bfloat>, ptr addrspace(3) %167, align 2
  %169 = shufflevector <2 x bfloat> %164, <2 x bfloat> %168, <2 x i32> <i32 0, i32 2>
  %170 = zext nneg i32 %158 to i64
  %171 = getelementptr inbounds [2 x i8], ptr addrspace(1) %4, i64 %170
  store <2 x bfloat> %169, ptr addrspace(1) %171, align 4
  %172 = shufflevector <2 x bfloat> %164, <2 x bfloat> %168, <2 x i32> <i32 1, i32 3>
  %173 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 256
  store <2 x bfloat> %172, ptr addrspace(1) %173, align 4
  %174 = add nuw nsw i32 %159, 8
  %175 = and i32 %174, 62
  %176 = zext nneg i32 %175 to i64
  %177 = zext nneg i32 %150 to i64
  %178 = add i64 %176, %177
  %179 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %178
  %180 = getelementptr inbounds i8, ptr addrspace(3) %179, i64 128
  %181 = load <2 x bfloat>, ptr addrspace(3) %180, align 2
  %182 = or disjoint i32 %175, %150
  %183 = zext nneg i32 %182 to i64
  %184 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %183
  %185 = load <2 x bfloat>, ptr addrspace(3) %184, align 2
  %186 = shufflevector <2 x bfloat> %185, <2 x bfloat> %181, <2 x i32> <i32 0, i32 2>
  %187 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 2048
  store <2 x bfloat> %186, ptr addrspace(1) %187, align 2
  %188 = shufflevector <2 x bfloat> %185, <2 x bfloat> %181, <2 x i32> <i32 1, i32 3>
  %189 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 2304
  store <2 x bfloat> %188, ptr addrspace(1) %189, align 2
  %190 = add i32 %174, 8
  %191 = and i32 %190, 62
  %192 = zext nneg i32 %191 to i64
  %193 = add i64 %192, %177
  %194 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %193
  %195 = getelementptr inbounds i8, ptr addrspace(3) %194, i64 128
  %196 = load <2 x bfloat>, ptr addrspace(3) %195, align 2
  %197 = or disjoint i32 %191, %150
  %198 = zext nneg i32 %197 to i64
  %199 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %198
  %200 = load <2 x bfloat>, ptr addrspace(3) %199, align 2
  %201 = shufflevector <2 x bfloat> %200, <2 x bfloat> %196, <2 x i32> <i32 0, i32 2>
  %202 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 4096
  store <2 x bfloat> %201, ptr addrspace(1) %202, align 2
  %203 = shufflevector <2 x bfloat> %200, <2 x bfloat> %196, <2 x i32> <i32 1, i32 3>
  %204 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 4352
  store <2 x bfloat> %203, ptr addrspace(1) %204, align 2
  %205 = add i32 %174, 16
  %206 = and i32 %205, 62
  %207 = zext nneg i32 %206 to i64
  %208 = add i64 %207, %177
  %209 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %208
  %210 = getelementptr inbounds i8, ptr addrspace(3) %209, i64 128
  %211 = load <2 x bfloat>, ptr addrspace(3) %210, align 2
  %212 = or disjoint i32 %206, %150
  %213 = zext nneg i32 %212 to i64
  %214 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %213
  %215 = load <2 x bfloat>, ptr addrspace(3) %214, align 2
  %216 = shufflevector <2 x bfloat> %215, <2 x bfloat> %211, <2 x i32> <i32 0, i32 2>
  %217 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 6144
  store <2 x bfloat> %216, ptr addrspace(1) %217, align 2
  %218 = shufflevector <2 x bfloat> %215, <2 x bfloat> %211, <2 x i32> <i32 1, i32 3>
  %219 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 6400
  store <2 x bfloat> %218, ptr addrspace(1) %219, align 2
  %220 = xor i32 %160, 32
  %221 = or disjoint i32 %220, %150
  %222 = zext nneg i32 %221 to i64
  %223 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %222
  %224 = load <2 x bfloat>, ptr addrspace(3) %223, align 2
  %225 = zext nneg i32 %220 to i64
  %226 = add i64 %225, %177
  %227 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %226
  %228 = getelementptr inbounds i8, ptr addrspace(3) %227, i64 128
  %229 = load <2 x bfloat>, ptr addrspace(3) %228, align 2
  %230 = shufflevector <2 x bfloat> %224, <2 x bfloat> %229, <2 x i32> <i32 0, i32 2>
  %231 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 8192
  store <2 x bfloat> %230, ptr addrspace(1) %231, align 2
  %232 = shufflevector <2 x bfloat> %224, <2 x bfloat> %229, <2 x i32> <i32 1, i32 3>
  %233 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 8448
  store <2 x bfloat> %232, ptr addrspace(1) %233, align 2
  %234 = add i32 %174, 32
  %235 = and i32 %234, 62
  %236 = zext nneg i32 %235 to i64
  %237 = add i64 %236, %177
  %238 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %237
  %239 = getelementptr inbounds i8, ptr addrspace(3) %238, i64 128
  %240 = load <2 x bfloat>, ptr addrspace(3) %239, align 2
  %241 = or disjoint i32 %235, %150
  %242 = zext nneg i32 %241 to i64
  %243 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %242
  %244 = load <2 x bfloat>, ptr addrspace(3) %243, align 2
  %245 = shufflevector <2 x bfloat> %244, <2 x bfloat> %240, <2 x i32> <i32 0, i32 2>
  %246 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 10240
  store <2 x bfloat> %245, ptr addrspace(1) %246, align 2
  %247 = shufflevector <2 x bfloat> %244, <2 x bfloat> %240, <2 x i32> <i32 1, i32 3>
  %248 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 10496
  store <2 x bfloat> %247, ptr addrspace(1) %248, align 2
  %249 = add i32 %174, 40
  %250 = and i32 %249, 62
  %251 = zext nneg i32 %250 to i64
  %252 = add i64 %251, %177
  %253 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %252
  %254 = getelementptr inbounds i8, ptr addrspace(3) %253, i64 128
  %255 = load <2 x bfloat>, ptr addrspace(3) %254, align 2
  %256 = or disjoint i32 %250, %150
  %257 = zext nneg i32 %256 to i64
  %258 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %257
  %259 = load <2 x bfloat>, ptr addrspace(3) %258, align 2
  %260 = shufflevector <2 x bfloat> %259, <2 x bfloat> %255, <2 x i32> <i32 0, i32 2>
  %261 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 12288
  store <2 x bfloat> %260, ptr addrspace(1) %261, align 2
  %262 = shufflevector <2 x bfloat> %259, <2 x bfloat> %255, <2 x i32> <i32 1, i32 3>
  %263 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 12544
  store <2 x bfloat> %262, ptr addrspace(1) %263, align 2
  %264 = add i32 %174, 48
  %265 = and i32 %264, 62
  %266 = zext nneg i32 %265 to i64
  %267 = add i64 %266, %177
  %268 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %267
  %269 = getelementptr inbounds i8, ptr addrspace(3) %268, i64 128
  %270 = load <2 x bfloat>, ptr addrspace(3) %269, align 2
  %271 = or disjoint i32 %265, %150
  %272 = zext nneg i32 %271 to i64
  %273 = getelementptr inbounds [2 x i8], ptr addrspace(3) @shared_0, i64 %272
  %274 = load <2 x bfloat>, ptr addrspace(3) %273, align 2
  %275 = shufflevector <2 x bfloat> %274, <2 x bfloat> %270, <2 x i32> <i32 0, i32 2>
  %276 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 14336
  store <2 x bfloat> %275, ptr addrspace(1) %276, align 2
  %277 = shufflevector <2 x bfloat> %274, <2 x bfloat> %270, <2 x i32> <i32 1, i32 3>
  %278 = getelementptr inbounds i8, ptr addrspace(1) %171, i64 14592
  store <2 x bfloat> %277, ptr addrspace(1) %278, align 2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_broadcast_2(ptr noalias readonly align 256 captures(none) dereferenceable(8388608) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(33554432) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = and i32 %8, 523776
  %11 = shl nuw nsw i32 %5, 7
  %12 = and i32 %11, 3670016
  %13 = or disjoint i32 %12, %10
  %14 = or disjoint i32 %13, %7
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %15
  %17 = load <4 x bfloat>, ptr addrspace(1) %16, align 8, !invariant.load !6
  %18 = zext nneg i32 %9 to i64
  %19 = getelementptr inbounds [2 x i8], ptr addrspace(1) %4, i64 %18
  store <4 x bfloat> %17, ptr addrspace(1) %19, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_transpose(ptr noalias readonly align 256 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(33554432) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = and i32 %7, 124
  %11 = shl i32 %5, 21
  %12 = and i32 %11, 14680064
  %13 = or disjoint i32 %10, %12
  %14 = shl nuw nsw i32 %6, 14
  %15 = and i32 %14, 1572864
  %16 = or disjoint i32 %13, %15
  %17 = shl nuw nsw i32 %5, 4
  %18 = and i32 %17, 524160
  %19 = or disjoint i32 %16, %18
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %20
  %22 = load <4 x bfloat>, ptr addrspace(1) %21, align 8, !invariant.load !6
  %23 = zext nneg i32 %9 to i64
  %24 = getelementptr inbounds [2 x i8], ptr addrspace(1) %4, i64 %23
  store <4 x bfloat> %22, ptr addrspace(1) %24, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_convert_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(67108864) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %10
  %12 = load <4 x bfloat>, ptr addrspace(1) %11, align 8, !invariant.load !6
  %13 = extractelement <4 x bfloat> %12, i32 0
  %14 = extractelement <4 x bfloat> %12, i32 1
  %15 = extractelement <4 x bfloat> %12, i32 2
  %16 = extractelement <4 x bfloat> %12, i32 3
  %17 = fpext bfloat %13 to float
  %18 = fpext bfloat %14 to float
  %19 = fpext bfloat %15 to float
  %20 = fpext bfloat %16 to float
  %21 = insertelement <4 x float> poison, float %17, i64 0
  %22 = insertelement <4 x float> %21, float %18, i64 1
  %23 = insertelement <4 x float> %22, float %19, i64 2
  %24 = insertelement <4 x float> %23, float %20, i64 3
  %25 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %10
  store <4 x float> %24, ptr addrspace(1) %25, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_convert(ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(67108864) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds [2 x i8], ptr addrspace(1) %3, i64 %10
  %12 = load <4 x bfloat>, ptr addrspace(1) %11, align 8, !invariant.load !6
  %13 = extractelement <4 x bfloat> %12, i32 0
  %14 = extractelement <4 x bfloat> %12, i32 1
  %15 = extractelement <4 x bfloat> %12, i32 2
  %16 = extractelement <4 x bfloat> %12, i32 3
  %17 = fpext bfloat %13 to float
  %18 = fpext bfloat %14 to float
  %19 = fpext bfloat %15 to float
  %20 = fpext bfloat %16 to float
  %21 = insertelement <4 x float> poison, float %17, i64 0
  %22 = insertelement <4 x float> %21, float %18, i64 1
  %23 = insertelement <4 x float> %22, float %19, i64 2
  %24 = insertelement <4 x float> %23, float %20, i64 3
  %25 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %10
  store <4 x float> %24, ptr addrspace(1) %25, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_add(ptr noalias align 256 captures(none) dereferenceable(67108864) %0, ptr noalias readonly align 256 captures(none) dereferenceable(67108864) %1, ptr noalias readnone align 256 captures(none) dereferenceable(67108864) %2) local_unnamed_addr #5 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %8 = shl nuw nsw i32 %7, 2
  %9 = shl nuw nsw i32 %6, 9
  %10 = or disjoint i32 %8, %9
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %11
  %13 = getelementptr inbounds [4 x i8], ptr addrspace(1) %5, i64 %11
  %14 = load <4 x float>, ptr addrspace(1) %13, align 16
  %15 = extractelement <4 x float> %14, i32 0
  %16 = extractelement <4 x float> %14, i32 1
  %17 = extractelement <4 x float> %14, i32 2
  %18 = extractelement <4 x float> %14, i32 3
  %19 = load <4 x float>, ptr addrspace(1) %12, align 16, !invariant.load !6
  %20 = extractelement <4 x float> %19, i32 0
  %21 = extractelement <4 x float> %19, i32 1
  %22 = extractelement <4 x float> %19, i32 2
  %23 = extractelement <4 x float> %19, i32 3
  %24 = fadd float %20, %15
  %25 = fadd float %21, %16
  %26 = fadd float %22, %17
  %27 = fadd float %23, %18
  %28 = insertelement <4 x float> poison, float %24, i64 0
  %29 = insertelement <4 x float> %28, float %25, i64 1
  %30 = insertelement <4 x float> %29, float %26, i64 2
  %31 = insertelement <4 x float> %30, float %27, i64 3
  store <4 x float> %31, ptr addrspace(1) %13, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_convert_3(ptr noalias readonly align 256 captures(none) dereferenceable(67108864) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(33554432) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds [4 x i8], ptr addrspace(1) %3, i64 %10
  %12 = load <4 x float>, ptr addrspace(1) %11, align 16, !invariant.load !6
  %13 = extractelement <4 x float> %12, i32 0
  %14 = extractelement <4 x float> %12, i32 1
  %15 = extractelement <4 x float> %12, i32 2
  %16 = extractelement <4 x float> %12, i32 3
  %17 = fptrunc float %13 to bfloat
  %18 = fptrunc float %14 to bfloat
  %19 = fptrunc float %15 to bfloat
  %20 = fptrunc float %16 to bfloat
  %21 = insertelement <4 x bfloat> poison, bfloat %17, i64 0
  %22 = insertelement <4 x bfloat> %21, bfloat %18, i64 1
  %23 = insertelement <4 x bfloat> %22, bfloat %19, i64 2
  %24 = insertelement <4 x bfloat> %23, bfloat %20, i64 3
  %25 = getelementptr inbounds [2 x i8], ptr addrspace(1) %4, i64 %10
  store <4 x bfloat> %24, ptr addrspace(1) %25, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.rsqrt.approx.f(float) #10

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.ex2.approx.f32(float) #12

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { convergent nocallback nofree nounwind memory(argmem: read) }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { nounwind "nvvm.reqntid"="128,1,1" }
attributes #5 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #6 = { nounwind "nvvm.reqntid"="64,1,1" }
attributes #7 = { nounwind }
attributes #8 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "nvvm.reqntid"="128,1,1" }
attributes #9 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #11 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #12 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!nvvm.annotations = !{}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 32768}
!5 = !{i32 0, i32 128}
!6 = !{}
!7 = !{i32 0, i32 32}
!8 = !{i32 0, i32 8192}
!9 = !{i32 0, i32 1048576}
!10 = !{i32 0, i32 1024}
