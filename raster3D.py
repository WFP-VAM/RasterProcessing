import os
import sys
import time
import tempfile
import shutil
import errno
import stat
import glob
import gdal
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from itertools import product
from functools import partial
from joblib import Parallel, delayed, load, dump
from gdalnumeric import BandWriteArray
from gdalconst import GA_ReadOnly
import pdb
import memory_profiler
import psutil
import argparse
import sharedmem


#---------------------------MAIN CODE (DO NOT TOUCH)------------------------------
def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise

def sliding_window(a,ws,ss = None,flatten = False):
    '''
    Return a sliding window over a in any number of dimensions
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.
    Returns
        an array containing each n-dimensional window from a
    '''

    def norm_shape(shape):
        '''
        Normalize numpy array shapes so they're always expressed as a tuple,
        even for one-dimensional shapes.
        Parameters: shape - an int, or a tuple of ints
        Returns: a shape tuple
        '''
        try:
            i = int(shape)

            return (i,)
        except TypeError:
            # shape was not a number
            pass
        try:
            t = tuple(shape)
            #print t
            return t
        except TypeError:
            # shape was not iterable
            pass
        raise TypeError('shape must be an int, or a tuple of ints')

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension. a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)

def slidingWindows(DATA,**kwargs):

    def calcWindow(DATA,**kwargs):
        windowSize = list(np.shape(DATA))
        stepSize = [0,0,0]
        if 'tLen' in kwargs.keys() or 'tStep' in kwargs.keys():
            if 'tLen' in kwargs.keys() and 'tStep' in kwargs.keys():
                windowSize[2] = kwargs['tLen']
                stepSize[2] = kwargs['tStep']
            else:
                print 'ERROR! input arguments incomplete'

        if 'xLen' in kwargs.keys() or 'xStep' in kwargs.keys():
            if 'xLen' in kwargs.keys() and 'xStep' in kwargs.keys():
                windowSize[0] = kwargs['xLen']
                stepSize[0] = kwargs['xStep']
            else:
                print 'ERROR! input arguments incomplete'

        if 'yLen' in kwargs.keys() or 'yStep' in kwargs.keys():
            if 'tLen' in kwargs.keys() and 'yStep' in kwargs.keys():
                windowSize[1] = kwargs['yLen']
                stepSize[1] = kwargs['yStep']
            else:
                print 'ERROR! input arguments incomplete'

        return windowSize, stepSize

    windowSize, stepSize = calcWindow(DATA,**kwargs)
    return sliding_window(DATA,windowSize,stepSize)

def rasterStack(rasterList):

    inRaster = gdal.Open(rasterList[0], GA_ReadOnly)
    array = inRaster.GetRasterBand(1).ReadAsArray()
    dim = np.shape(array)
    print len(rasterList)

    DATA = np.zeros((dim[0],dim[1],len(rasterList)),array.dtype,order='C')

    for i,rasterFile in enumerate(rasterList):
        inRaster = gdal.Open(rasterFile, GA_ReadOnly) # gdal.GA_ReadOnly
        DATA[:,:,i] = inRaster.GetRasterBand(1).ReadAsArray()

        print '---- Loaded '+rasterFile
        # if i==0:
        #     DATA = inRaster.GetRasterBand(1).ReadAsArray()
        # else:
        #     array = inRaster.GetRasterBand(1).ReadAsArray()
        #     DATA = np.dstack((DATA,array))

    return DATA

def readFromDir(inptFolder,start='',end='', dek=''):

    def findRasters(inptFolder,start='',end='', dek=''):
        fileList = glob.glob(inptFolder+'\*.rst')+glob.glob(inptFolder+'\*.tif')+ \
                    glob.glob(inptFolder+'\*.tiff')+glob.glob(inptFolder+'\*.mpg')

        prefix = os.path.basename(os.path.commonprefix(fileList))[:6]

      #  first_file = [x for x in fileList if x[-8:-4] == dek][0]

      #  year = first_file[len(inptFolder)+2+5:len(inptFolder)+2+9]

        if start!='':
            start_file = inptFolder+'\\' + prefix+start+'.rst'
            index_start = fileList.index(start_file)
            fileList = fileList[index_start:]
            year = fileList[0][len(inptFolder)+2+5:len(inptFolder)+2+9]

        if end!='':
            end_file = inptFolder+'\\'+prefix+end+'.rst'
            index_end = fileList.index(end_file)
            fileList = fileList[:index_end+1]

        if dek!='':

            dek_file = inptFolder+'\\'+prefix+year+dek+'.rst'
            index_dek = fileList.index(dek_file)
            fileList = filter(lambda dek_file: dek == dek_file[-8:-4], fileList)

        return fileList

    start_time = time.time()

    fileList = findRasters(inptFolder,start,end,dek)
    namesList = np.array(fileList).reshape(1,1,len(fileList))
    DATA = rasterStack(fileList)

    print("---Files Loaded in %s seconds ---" % (time.time() - start_time))
    return DATA, namesList

def readFromFileList(fileListFile):

    f = open(fileListFile, 'r')
    fileList = f.readlines()
    print fileList

    namesList = np.array(fileList).reshape(1,1,len(fileList))
    DATA = rasterStack(fileList)

    print("---Files Loaded in %s seconds ---" % (time.time() - start_time))
    return DATA, namesList

def writeRaster(raster2D,outputFile,templateFile):
    inRaster = gdal.Open(templateFile, GA_ReadOnly)
    inDriver = inRaster.GetDriver()
    outRaster = inDriver.CreateCopy(outputFile, inRaster, 0)
    outBand = outRaster.GetRasterBand(1)
    test_out = BandWriteArray(outBand,raster2D,0,0)
    #close  the dataset
    inRaster = None
    outRaster = None
    print '--- '+outputFile+' created----'

    return True

def writeRstrStack(DATA,fileList,outputDir='',prefix='',suffix='', **kwargs):
    if kwargs!={}:
        template3D = slidingWindows(fileList,**kwargs)
    else:
        template3D = fileList

    dim = np.shape(DATA)
    for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2])):
        templateFile =  template3D[i].flatten().tolist()[-1]
        if outputDir=='':
            outputDir = os.path.split(templateFile)[0]

        base, ext = os.path.splitext(os.path.basename(templateFile))
        # base, ext = os.path.splitext(os.path.basename(templateFile))

        if suffix =='':
            outputFile = prefix+base[6:]+suffix+ext
        else:
            #suffix = base[6:10]
            outputFile = base[0:6]+suffix+base[10:14]+ext

        outputFile = os.path.join(outputDir,outputFile)

        writeRaster(DATA[i],outputFile,templateFile)

    return True

def memoryMap(DATA):
    # Dump the input data to disk to free the memory
    tmpDir = os.path.join('C:\\GeoSpatial\\temp\\rstrProc',time.strftime('%y%m%d-%H%M%S',time.gmtime()))
    os.makedirs(tmpDir)
    inputDir = os.path.join(tmpDir, 'input')
    dump(DATA,inputDir)
    # Release the reference on the original in memory array and replace it by a reference to the memmap array so that the garbage collector releases the memory
    DATA = load(inputDir, mmap_mode='r')
    return DATA, tmpDir

def applySrl(DATA,func,*args,**kwargs):
    start_time = time.time()

    dim = np.shape(DATA)
    if len(dim)>3:
        k = 0
        trgtFunc = closure(func,*args,**kwargs)
        for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2])):
            RSLT = trgtFunc(DATA[i])
            if k==0:
                RSLTstack = RSLT
            else:
                RSLTstack = np.dstack((RSLTstack,RSLT))

            k+=1
    else:
        RSLTstack = func(DATA,*args,**kwargs)

    print("---Serial Processed in %s seconds ---" % (time.time() - start_time))
    return RSLTstack
#---------------------------MAIN CODE (DO NOT TOUCH)------------------------------

#-------------FUNCTION OF INTEREST----------------
### ALL FUNCTIONS MUST USE ARGUMENTS DATA, RSLT, i
##def rasterFunc(DATA,RSLT,i):
##    RSLT[i] = np.sum(DATA[i],axis=2)
##    return True

def rasterFunc(DATA,RSLT,i,threshold):
    RSLT[i] = np.sum(DATA[i] > threshold,axis=2)
    return True

#rain days:     # count the number or elements (raindays) greater than the threshold
##def calc_rainfall(DATA,RSLT,i, thresh):
##    RSLT[i] = (DATA[i] > thresh).sum()
##    return True


###rain days
##def calc_rainfall(DATA,RSLT,i, thresh):
##	RSLT[i] = np.sum((np.where(DATA[i]>thresh)),axis=2)
##	return True


###rain days
##def calc_rainfall(DATA,RSLT,i, thresh):
##	RSLT[i] = np.sum((np.where(DATA[i]>thresh)),axis=2)
##	return True
#-------------------------------------------------

if __name__ == '__main__':

    #-------------INPUT PARAMETERS----------------
    inptFolder = r'E:\WORKSarah\python\tst'
    inptMatch = ''
    otptFolder = r'E:\WORKSarah\python\tst\out'
    start= ''#'198108d3'
    end = ''#'200001d2'
    dek =''

    tStep = 10
    tLen = 30
    xStep = np.nan
    xLen = np.nan
    yStep = np.nan
    yLen = np.nan
    nJobs = 4 #of processors
    prefix = 'sdncn1'
    thresholds = [0]#0,1,2,5]
   #suffix = '_avg'
    #dek = '_sum'

    #---------------------------------------------

    t0 = time.time()
    DATA, fileList = readFromDir(inptFolder,start,end,dek)


    if True: #len(dim)>3:
        print "Beginning parallelization pre-process."
        for threshold in thresholds:
            print threshold
            try:

                DATA, tmpDir = memoryMap(DATA)
                # Modify slidingWindows input if windowing on other dimensions (x,y)
                DATA = slidingWindows(DATA,tStep=tStep,tLen=tLen)
                dim = np.shape(DATA)

                # Pre-allocate a writeable shared memory map as a container for the results of the parallel computation
                resultDir = os.path.join(tmpDir, 'result_' + str(threshold) +'_mm')
                RSLT = np.memmap(resultDir, dtype=DATA.dtype,shape=(dim[0],dim[1],dim[2],dim[3],dim[4]), mode='w+')

                # Fork the worker processes to perform computation concurrently
                print 'Finished pre-process. Parallel processing:'
                start_time = time.time()
##                #Modify delayed(~) with function name that you are interested in
##                Parallel(n_jobs=nJobs)(delayed(rasterFunc(DATA, RSLT, i) for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2]))))
##                print("---Parallel Processed in %s seconds ---" % (time.time() - start_time))

                #Modify delayed(~) with function name that you are interested in
                Parallel(n_jobs=nJobs)(delayed(rasterFunc)(DATA, RSLT, i,threshold) for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2])))
                print("---Parallel Processed in %s seconds ---" % (time.time() - start_time))


            except Exception,e:
                print e
                pdb.set_trace()

        print 'Writing Results to Raster Files'
       # print fileList
        # Modify slidingWindows input if windowing on other dimensions (x,y) or add outputDir=
        #writeRstrStack(RSLT,fileList,outputDir=otptFolder,prefix=prefix,dek=dek,tStep=tStep,tLen=tLen)
        #writeRstrStack(RSLT,fileList,outputDir=otptFolder,suffix=suffix,prefix=prefix,tStep=tStep,tLen=tLen)
        writeRstrStack(RSLT,fileList,outputDir=otptFolder,prefix=prefix,tStep=tStep,tLen=tLen)
        #writeRstrStack(RSLT,fileList,outputDir=otptFolder,prefix=prefix,suffix=suffix,tStep=tStep,tLen=tLen)


        #writeRstrStack(DATA,fileList,outputDir='',prefix='',suffix='',**kwargs)


        print("---Program finished in %s seconds ---" % (time.time() - t0))

        try:
            shutil.rmtree(tmpDir, ignore_errors=False, onerror=handleRemoveReadonly)
        except Exception,e:
            print 'DELETE THE '+tmpDir+' TEMP FOLDER IN GEOSPATIAL!!!!!!!!!!!!!!!!!!!!!'
            print e
            pdb.set_trace()

#DATA = sharedmem.copy(DATA)#mapFunc = partial(func,*args,**kwargs)
# trgtFunc = closure(DATA,np.mean,axis=2)
# #pool = mp.Pool(processes=max(mp.cpu_count()-1,2),initargs=(DATA,))
# pdb.set_trace()
# RSLT = parmap(trgtFunc,[i for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2]))])
# #RSLT = pool.map(trgtFunc, [i for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2]))])
# #pool.close()
# #pool.join()
# pdb.set_trace()

# dim = np.shape(DATA)
# if len(dim)>3:
#     DATA = sharedmem.copy(DATA)#mapFunc = partial(func,*args,**kwargs)
#     trgtFunc = closure(DATA,np.mean,axis=2)
#     #pool = mp.Pool(processes=max(mp.cpu_count()-1,2),initargs=(DATA,))
#     pdb.set_trace()
#     RSLT = parmap(trgtFunc,[i for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2]))])
#     #RSLT = pool.map(trgtFunc, [i for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2]))])
#     #pool.close()
#     #pool.join()
#     pdb.set_trace()
# else:
#     pdb.set_trace()
#     RSLT = func(DATA,*args,**kwargs)

#    def closure(func,*args,**kwargs):
#     if len(args)==0 and len(kwargs)==0:
#         def mapFunc(x):
#             return func(x)
#         return mapFunc
#     elif len(args)==0 and len(kwargs)>0:
#         def mapFunc(x):
#             return func(x,**kwargs)
#         return mapFunc
#     elif len(args)>0 and len(kwargs)==0:
#         def mapFunc(x):
#             return func(x,*args)
#         return mapFunc
#     else:
#         def mapFunc(x):
#             return func(x,*args,**kwargs)
#         return mapFunc

# def fun(f,q_in,q_out):
#     while True:
#         i,x = q_in.get()
#         if i is None:
#             break
#         q_out.put((i,f(x)))

# def parmap(f, X, DATA=None, nprocs = max(mp.cpu_count()-1,2)):
#     q_in   = mp.Queue(1)
#     q_out  = mp.Queue()

#     if DATA!=None:
#         DATA = sharedmem.copy(DATA)
#         proc = [mp.Process(target=fun,args=(f,q_in,q_out,DATA)) for _ in range(nprocs)]
#     else:
#         proc = [mp.Process(target=fun,args=(f,q_in,q_out)) for _ in range(nprocs)]
#     for p in proc:
#         p.daemon = True
#         p.start()

#     sent = [q_in.put((i,x)) for i,x in enumerate(X)]
#     [q_in.put((None,None)) for _ in range(nprocs)]
#     res = [q_out.get() for _ in range(len(sent))]

#     [p.join() for p in proc]

#     return [x for i,x in sorted(res)]


# def apply3D(DATA,func,*args,**kwargs):

#     dim = np.shape(DATA)

#     if len(dim)>3:
#         DATA = sharedmem.copy(DATA)
#         trgtFunc = partial(func,*args,**kwargs)
#         #trgtFunc = closure(DATA,func,*args,**kwargs)
#         pool = mp.Pool(processes=max(mp.cpu_count()-1,2))
#         #pdb.set_trace()
#         #RSLT = parmap(lambda i: np.mean(DATA[i],axis=2),[i for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2]))])
#         RSLT = pool.map(trgtFunc, [DATA[i] for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2]))])
#         pool.close()
#         pool.join()
#         pdb.set_trace()
#     else:
#         pdb.set_trace()
#         RSLT = func(DATA,*args,**kwargs)

#     return RSLT

# def applyPrl(DATA,func):
#     dim = np.shape(DATA)
#     if len(dim)>3:
#         # trgtFunc = closure(func,*args,**kwargs)
#         parallelizer = Parallel(n_jobs=dim[0]*dim[1]*dim[2])
#         # this iterator returns the functions to execute for each task
#         tasks_iterator = ( delayed(func)(x) for x in [DATA[i] for i in product(np.arange(dim[0]),np.arange(dim[1]),np.arange(dim[2]))] )
#         result = parallelizer( tasks_iterator )
#         # merging the output of the jobs
#         return np.dstack(result)
#     else:
#         return func(DATA,*args,**kwargs)