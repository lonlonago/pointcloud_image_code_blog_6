#  TensorRT installation 

 First, you need to ensure that CUDA is installed correctly. After installation, verify whether it is installed through nvcc -V. 

 Download TensorRT URL: https://developer.nvidia.com/nvidia-tensorrt-8x-download, download the latest version to decompress 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957377234
  ```  
 In order to save the memory of the root directory, I put TensorRT under home and add environment variables. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957377234
  ```  
 Then, we use the sample program to determine whether TRT works normally. We first compile the sampleMNIST source code, and then generate an executable file in the bin directory. We switch to execute it directly. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957377234
  ```  
 ![avatar]( cab241af9af14f11bcdbd6a675051d33.png) 

  If the output is as follows, PASSED is displayed at the end, indicating that the sample passed. 

#  Python support 

 In the previous section, although we have installed TensorRT, our Python environment cannot be imported through import tensorrt, so we need to install the corresponding .whl. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957377234
  ```  
 ![avatar]( b6fe0c74fdaa4a90933005500887c74c.png) 

#  ONNX deployment 

 TensorRT is an inference optimization library accelerated by nvidia for nvidia graphics card training models on specific platforms. It is a C ++ library that only supports inference and does not support training; 

 For reasoning, you need to create an IExecutionContext object first. To create this object, you need to create an ICudaEngine object (engine) first. There are two ways to create an engine: 

##  C++ 

 The TensorRT version has changed a lot, you can directly check the API documentation 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957377234
  ```  
 CMakeLists.txt 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957377234
  ```  
##  python 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957377234
  ```  
##  tretexec 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957377234
  ```  
