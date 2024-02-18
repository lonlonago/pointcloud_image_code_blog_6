##  Step 1 Download the file 

  Baidu Netdisk: Link: https://pan.baidu.com/s/1WQQ8kaDilaagjoK5IrYZzA Extraction code: 1111  

  Note: Unzip it directly on the E disk!!!! You can do it without unzipping it on the E disk. Just replace the environment variables and the address in the attribute table file (props file) later.   

##   2. Configuring environment variables 

  Click Computer, Settings, Search and Edit System environment variables Click Path to add the following variables E:\ PCL1.11.0\ bin E:\ PCL1.11.0\ 3rdParty\ VTK\ bin E:\ PCL1.11.0\ 3rdParty\ OpenNI2\ Redist E:\ PCL1.11.0\ 3rdParty\ FLANN\ bin 

##   3. Visual Studio Configuration Property Sheet 

  1. Create blank C++ new project 

 Note: Change x86 to x64. 

 Add Property Sheet, View - > Other Windows - > Explorer, right-click Debug | x64- > Add Existing Property Sheet - > Add pcl1_11_x64_debug .props (in the downloaded and decompressed folder) Right-click Release | x64- > Add Existing Property Sheet - > Add pcl1_11_x64_release .props (in the downloaded and decompressed folder)   

##  4 tests 

  Click Solution Explorer - > Right-click Source File - > Add - > New Item, add c ++ file 

 Paste test code 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573799485
  ```  
  The program generates an elliptical cylindrical point cloud and colours it along the axis 

