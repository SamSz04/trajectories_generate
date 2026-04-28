; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_convert_3(ptr noalias align 256 dereferenceable(67108864) %0, ptr noalias align 256 dereferenceable(33554432) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = mul i32 %4, 4
  %6 = mul i32 %3, 512
  %7 = add i32 %5, %6
  %8 = getelementptr inbounds [16777216 x float], ptr %0, i32 0, i32 %7
  %9 = load <4 x float>, ptr %8, align 4, !invariant.load !3
  %10 = extractelement <4 x float> %9, i64 0
  %11 = fptrunc float %10 to bfloat
  %12 = extractelement <4 x float> %9, i64 1
  %13 = fptrunc float %12 to bfloat
  %14 = extractelement <4 x float> %9, i64 2
  %15 = fptrunc float %14 to bfloat
  %16 = extractelement <4 x float> %9, i64 3
  %17 = fptrunc float %16 to bfloat
  %18 = insertelement <4 x bfloat> poison, bfloat %11, i32 0
  %19 = insertelement <4 x bfloat> %18, bfloat %13, i32 1
  %20 = insertelement <4 x bfloat> %19, bfloat %15, i32 2
  %21 = insertelement <4 x bfloat> %20, bfloat %17, i32 3
  %22 = getelementptr inbounds [16777216 x bfloat], ptr %1, i32 0, i32 %7
  store <4 x bfloat> %21, ptr %22, align 2
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32768}
!2 = !{i32 0, i32 128}
!3 = !{}
