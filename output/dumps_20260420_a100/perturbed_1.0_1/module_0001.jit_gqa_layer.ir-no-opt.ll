; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_convert_fusion_1(ptr noalias align 256 dereferenceable(67108864) %0, ptr noalias align 16 dereferenceable(33554432) %1, ptr noalias align 256 dereferenceable(33554432) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %5, 4
  %7 = mul i32 %4, 512
  %8 = add i32 %6, %7
  %9 = getelementptr inbounds [16777216 x float], ptr %0, i32 0, i32 %8
  %10 = load <4 x float>, ptr %9, align 4, !invariant.load !3
  %11 = getelementptr inbounds [16777216 x bfloat], ptr %1, i32 0, i32 %8
  %12 = load <4 x bfloat>, ptr %11, align 2, !invariant.load !3
  %13 = extractelement <4 x bfloat> %12, i64 0
  %14 = fpext bfloat %13 to float
  %15 = extractelement <4 x float> %10, i64 0
  %16 = fadd float %14, %15
  %17 = fptrunc float %16 to bfloat
  %18 = extractelement <4 x bfloat> %12, i64 1
  %19 = fpext bfloat %18 to float
  %20 = extractelement <4 x float> %10, i64 1
  %21 = fadd float %19, %20
  %22 = fptrunc float %21 to bfloat
  %23 = extractelement <4 x bfloat> %12, i64 2
  %24 = fpext bfloat %23 to float
  %25 = extractelement <4 x float> %10, i64 2
  %26 = fadd float %24, %25
  %27 = fptrunc float %26 to bfloat
  %28 = extractelement <4 x bfloat> %12, i64 3
  %29 = fpext bfloat %28 to float
  %30 = extractelement <4 x float> %10, i64 3
  %31 = fadd float %29, %30
  %32 = fptrunc float %31 to bfloat
  %33 = insertelement <4 x bfloat> poison, bfloat %17, i32 0
  %34 = insertelement <4 x bfloat> %33, bfloat %22, i32 1
  %35 = insertelement <4 x bfloat> %34, bfloat %27, i32 2
  %36 = insertelement <4 x bfloat> %35, bfloat %32, i32 3
  %37 = getelementptr inbounds [16777216 x bfloat], ptr %2, i32 0, i32 %8
  store <4 x bfloat> %36, ptr %37, align 2
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
