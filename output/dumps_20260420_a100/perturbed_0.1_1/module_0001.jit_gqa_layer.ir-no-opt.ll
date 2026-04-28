; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_convert_fusion(ptr noalias align 16 dereferenceable(33554432) %0, ptr noalias align 256 dereferenceable(33554432) %1, ptr noalias align 256 dereferenceable(33554432) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %5, 4
  %7 = mul i32 %4, 512
  %8 = add i32 %6, %7
  %9 = getelementptr inbounds [16777216 x bfloat], ptr %1, i32 0, i32 %8
  %10 = load <4 x bfloat>, ptr %9, align 2
  %11 = getelementptr inbounds [16777216 x bfloat], ptr %0, i32 0, i32 %8
  %12 = load <4 x bfloat>, ptr %11, align 2, !invariant.load !3
  %13 = extractelement <4 x bfloat> %12, i64 0
  %14 = extractelement <4 x bfloat> %10, i64 0
  %15 = fpext bfloat %13 to float
  %16 = fpext bfloat %14 to float
  %17 = fadd float %15, %16
  %18 = fptrunc float %17 to bfloat
  %19 = extractelement <4 x bfloat> %12, i64 1
  %20 = extractelement <4 x bfloat> %10, i64 1
  %21 = fpext bfloat %19 to float
  %22 = fpext bfloat %20 to float
  %23 = fadd float %21, %22
  %24 = fptrunc float %23 to bfloat
  %25 = extractelement <4 x bfloat> %12, i64 2
  %26 = extractelement <4 x bfloat> %10, i64 2
  %27 = fpext bfloat %25 to float
  %28 = fpext bfloat %26 to float
  %29 = fadd float %27, %28
  %30 = fptrunc float %29 to bfloat
  %31 = extractelement <4 x bfloat> %12, i64 3
  %32 = extractelement <4 x bfloat> %10, i64 3
  %33 = fpext bfloat %31 to float
  %34 = fpext bfloat %32 to float
  %35 = fadd float %33, %34
  %36 = fptrunc float %35 to bfloat
  %37 = insertelement <4 x bfloat> poison, bfloat %18, i32 0
  %38 = insertelement <4 x bfloat> %37, bfloat %24, i32 1
  %39 = insertelement <4 x bfloat> %38, bfloat %30, i32 2
  %40 = insertelement <4 x bfloat> %39, bfloat %36, i32 3
  store <4 x bfloat> %40, ptr %9, align 2
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
