import os
'''重新编译 基于C++的so库'''

maskrcnn_source_path = os.path.abspath("./source/maskrcnn-benchmark/")
darknet_source_path = os.path.abspath("./source/darknet/")

maskrcnn_target_path = os.path.abspath("./maskrcnn_benchmark/")
darknet_target_path = os.path.abspath("./darknet/")

# setup maskrcnn-benchmark

os.system("rm -r %s" % os.path.join(maskrcnn_target_path, "maskrcnn_benchmark/_C.*"))

os.system("cd %s && python setup.py build develop" % maskrcnn_source_path)

os.system("cp %s %s" % (os.path.join(maskrcnn_source_path, "maskrcnn_benchmark", "_C.*"), os.path.join(maskrcnn_target_path)))


# setup darknet
os.system("cd %s && make clean && make -j8" % darknet_source_path)

os.system("cd %s && cp libdarknet.a libdarknet.so darknet %s" % (darknet_source_path, darknet_target_path))
os.system("cd %s && make clean" % darknet_source_path)
