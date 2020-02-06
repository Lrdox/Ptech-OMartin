import os
import tempfile
import zipfile

_,zip1 = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip1, 'w', compression=zipfile.ZIP_DEFLATED) as f :
    f.write('../test.tflite')
print("size of the unpruned model: %.2f Mb" %(os.path.getsize('alexnet/sgd_custom/alexnet.h5') / float(2**20)))
print("size of the pruned model before compression: %.2f Mb" %(os.path.getsize('alexnet/pruned_sgd_custom/sparsity95/cifar10_pruned_alexnet.h5') / float(2**20)))

print("size of the pruned model after compression: %.2f Mb" %(os.path.getsize(zip1) / float(2**20)))



#simplenetslim/pruned_sgd_custom/sparsity95/cifar10_pruned_simplenetslim.h5
#simplenet/pruned_sgd_custom/sparsity90/cifar10_pruned_simplenet.h5
#alexnet/pruned_sgd_custom/sparsity90/cifar10_pruned_alexnet.h5