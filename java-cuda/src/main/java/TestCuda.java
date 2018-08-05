import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

public class TestCuda {

    public static void main(String[] args){

//        Pointer pointer = new Pointer();
//        cudaMalloc(pointer, 40);
//        System.out.println("Pointer: " + pointer);
//        cudaFree(pointer);


        JCudaDriver.setExceptionsEnabled(true);
        String ptxFileName = "kernels/testAddKernel.ptx";

        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module,"add");

        int numElements = 100000;

        float hostInputA[] = new float[numElements];
        float hostInputB[] = new float[numElements];

        for(int i = 0; i < numElements; i++){
            hostInputA[i] = (float)(i);
            hostInputB[i] = (float)(i);
        }


        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, numElements * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA), numElements * Sizeof.FLOAT);

        CUdeviceptr deviceInputB = new CUdeviceptr();
        cuMemAlloc(deviceInputB, numElements * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputA), numElements * Sizeof.FLOAT);


        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, numElements * Sizeof.FLOAT);

        Pointer kernelParemeters = Pointer.to(
                Pointer.to(new int[]{numElements}),
                Pointer.to(deviceInputB),
                Pointer.to(deviceInputB),
                Pointer.to(deviceOutput)
        );

        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double) numElements/ blockSizeX);
        cuLaunchKernel(
                function,
                gridSizeX, 1, 1,
                blockSizeX, 1, 1,
                0, null,
                kernelParemeters, null
        );

        cuCtxSynchronize();

        float hostOutput[] = new float[numElements];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, numElements*Sizeof.FLOAT);

        boolean passed = true;
        for(int i = 0; i < numElements; i++){
            float expected = i+i;
            if(Math.abs(hostOutput[i] - expected) > 1e-5){
                System.out.println("At index " + i + " found" + hostOutput[i] + " but expected " + expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));

        cuMemFree(deviceInputA);
        cuMemFree(deviceInputB);
        cuMemFree(deviceOutput);




    }

}
