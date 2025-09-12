
# llvm搭建 
 注意: 该搭建逻辑是基于llvm源码就行搭建. 并非是基于vcpkg管理的逻辑. 
     windows上要使用  powershell
 
# 第一步
``` bash  
    
 wget  https://github.com/llvm/llvm-project/archive/cd708029e0b2869e80abe31ddb175f7c35361f90.tar.gz
 
 tar -xvzf cd708029e0b2869e80abe31ddb175f7c35361f90.tar.gz
 mv llvm-project-cd708029e0b2869e80abe31ddb175f7c35361f90  cd708029e0b2869e80abe31ddb175f7c35361f90

 cd cd708029e0b2869e80abe31ddb175f7c35361f90
 
 touch BUILD.bazel
 
 mkdir build
 
 cd build 
 
 cmake ../llvm -DCMAKE_BUILD_TYPE=Debug 

 
 cd ..
 
 cp ./build/include/llvm/Config/*.h  ./llvm/include/llvm/Config
 
 
 
```


第二步
进入到 bazel_proejct2文件夹根目录
假设 llvm_dir = D:\\repo\\llvm_repo\\cd708029e0b2869e80abe31ddb175f7c35361f90

```bash 
  
 
   python ./third_party/patches/uitls.py  --src  ./third_party/patches/cd708029e0b2869e80abe31ddb175f7c35361f90    --to  llvm_dir  
   
#   cd llvm_dir
#   
#   python gast_rename_bazel.py  
    
```
