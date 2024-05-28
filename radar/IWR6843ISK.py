import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import math
import matplotlib.pyplot as plt
#from drawnow import *

from scipy.fft import fft, fftfreq, ifft

# Change the configuration file name
configFileName = 'IWR6843ISK_profile_Azimuth.cfg'

CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0;
magnitude = np.zeros((8,256), dtype=np.float32) 
frame_fft = np.zeros((0,64,15), dtype=complex) 
frame_fft_no = np.zeros((0,64,15), dtype=complex)
frame_comp = np.zeros((0,8,15), dtype=complex) 
magnitude_phase = [[] for i in range(8)]


frameNumber = 0

#空串列 (適合搭配append)
#ant = [[] for i in range(8)] #list_2d []*8
#ant_phase = [[] for i in range(8)]

#list_3d = [ [[] for i in range(3)] for i in range(8)] # []*3*8
#list_3d_zero = [ [0]*3 for i in range(8)]  #[0,0,0]*8

# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports
    
    # Raspberry pi
    #CLIport = serial.Serial('/dev/ttyACM0', 115200)
    #Dataport = serial.Serial('/dev/ttyACM1', 921600)
    
    # Windows
    CLIport = serial.Serial('COM4', 115200)
    Dataport = serial.Serial('COM5', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)
        
    return CLIport, Dataport

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1;
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;
                
            digOutSampleRate = int(splitWords[11]);
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            
            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = float(splitWords[5]);

            
    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    configParameters["frametime"]=framePeriodicity*1e-3
    
    return configParameters
   
# ------------------------------------------------------------------

# Funtion to read and parse the incoming data
def readAndParseData18xx(Dataport, configParameters):
    global byteBuffer, byteBufferLength,frameNumber
    

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12;
    BYTE_VEC_ACC_MAX_SIZE = 2**15;
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
    MMWDEMO_UART_MSG_RANGE_PROFILE   = 2;
    MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4;
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5;
    maxBufferSize = 2**15;
    tlvHeaderLengthInBytes = 8;
    pointLengthInBytes = 16;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
    
    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
    #frameNumber = 0
    detObj = {}
    
    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8')
    byteCount = len(byteVec)
    
    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount
        
    # Check that the buffer has some data
    if byteBufferLength > 16:
        
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+8]
            if np.all(check == magicWord):
                startIdx.append(loc)
               
        # Check that startIdx is not empty
        if startIdx:
            
            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
                byteBufferLength = byteBufferLength - startIdx[0]
                
            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
                
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            
            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12+4],word)
            
            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1
    
    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2**8, 2**16, 2**24]
        
        # Initialize the pointer index
        idX = 0
        
        # Read the header
        magicNumber = byteBuffer[idX:idX+8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX+4],word)
        print(numDetectedObj)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX+4],word)
        print('numTLVs = ', numTLVs)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4

        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX:idX+4],word)
            idX += 4
            print(tlv_type)
            tlv_length = np.matmul(byteBuffer[idX:idX+4],word)
            print(tlv_length)
            idX += 4
                            
            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
            
                # Initialize the arrays
                x = np.zeros(numDetectedObj,dtype=np.float32)
                y = np.zeros(numDetectedObj,dtype=np.float32)
                z = np.zeros(numDetectedObj,dtype=np.float32)
                velocity = np.zeros(numDetectedObj,dtype=np.float32)
                
                for objectNum in range(numDetectedObj):
                    
                    # Read the data for each object
                    x[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                    y[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                    z[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                    velocity[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                
                # Store the data in the detObj dictionary
                detObj = {"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity":velocity}
                dataOK = 1
                        
            elif tlv_type == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP: 
  
                # Initialize the AZIMUT maps
                global frame_comp                                             
                comp = np.zeros((1,8,15), dtype=complex) 
                for i in range(8): #8ant
                    for objectNum in range(15): #20rangebin
                        imag = np.zeros(1,dtype=np.int16)
                        real = np.zeros(1,dtype=np.int16)                        
                        imag = byteBuffer[idX:idX+2].view(dtype=np.int16)
                        real = byteBuffer[idX+2:idX+4].view(dtype=np.int16)
                        temp = complex(float(real[0]),float(imag[0]))
                        comp[0][i][objectNum] = temp

                        idX += 4 
                                                  
                frame_comp = np.append(frame_comp,comp,axis=0)                    
                print(frame_comp.shape)
            else :
               idX = idX + tlv_length
                 
        # Remove already processed data
        if idX > 0 and byteBufferLength>idX:
            shiftSize = totalPacketLen
                            
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8')
            byteBufferLength = byteBufferLength - shiftSize
            
            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0         

    return dataOK, frameNumber, detObj

# ------------------------------------------------------------------

# Funtion to update the data and display in the plot
def update():
    
    dataOk = 0
    global detObj
    x = []
    y = []
      
    # Read and parse the received data
    dataOk, frameNumber, detObj = readAndParseData18xx(Dataport, configParameters)
    print('frameNumber = ',frameNumber)
    
    if dataOk and len(detObj["x"])>0 :  #
        #print(detObj)
        x = -detObj["x"]
        y = detObj["y"]
        
        #s.setData(x,y)
        #s.setData(magnitude_phase[0]) 
        #QtGui.QApplication.processEvents()
       
    
    return dataOk


# -------------------------    MAIN   -----------------------------------------  

# Configurate the serial port
CLIport, Dataport = serialConfig(configFileName)

# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

plt.close('all')

# Main loop 
detObj = {}  
frameData = {}    
currentIndex = 0
while True:
    try:
        # Update the data and check if the data is okay
        dataOk = update()
        
        if dataOk:
            # Store the current frame into frameData
            frameData[currentIndex] = detObj
            currentIndex += 1

        time.sleep(0.05) # Sampling frequency of 30 Hz 0.05
    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
                       
        #將每個frame的rangeAzimuthHeatMap做Angle-FFT
        frame_len = len(frame_comp)
        delta_t = configParameters["frametime"]
        timeaxis=np.arange(0, frame_len*delta_t, delta_t)     
        print('len = ',frame_len)
        frame_z = np.zeros((0,64,15), dtype=complex)
        frame_abs = np.zeros((0,64,15), dtype=np.float32)
        frame_phase = np.zeros((0,64,15), dtype=np.float32)    
                                    
        for i in range(frame_len): #range-theta           
            frame_temp = np.zeros((1,64,15), dtype=complex)
            frame_temp1 = np.zeros((1,64,15), dtype=np.float32)
            frame_temp2 = np.zeros((1,64,15), dtype=np.float32)

            for j in range(15):        
                padded_length = 64  
                frame_array = np.squeeze(frame_comp[i:i+1,:,j:j+1],axis=(0,2))   
                padded_signal = np.pad(frame_array, (0, padded_length - len(frame_array)), 'constant') #零填充
                fft_result = np.fft.fft(padded_signal) #complex
                fft_amplitude = np.abs(fft_result) #mag
                fft_phase = np.angle(fft_result) #phase
                                                          
                
                #變換位置
                middle_index = len(fft_amplitude) // 2
                fft_result_change = np.concatenate((fft_result[middle_index:], fft_result[:middle_index]))
                fft_amplitude_change = np.concatenate((fft_amplitude[middle_index:], fft_amplitude[:middle_index]))
                fft_phase_change = np.concatenate((fft_phase[middle_index:], fft_phase[:middle_index]))
                fft_result_change = np.expand_dims(fft_result_change,axis=0) #axis=1
                fft_result_change = np.expand_dims(fft_result_change,axis=2) #axis=0
                fft_amplitude_change = np.expand_dims(fft_amplitude_change,axis=0) #axis=1
                fft_amplitude_change = np.expand_dims(fft_amplitude_change,axis=2) #axis=0    
                fft_phase_change = np.expand_dims(fft_phase_change,axis=0) #axis=1
                fft_phase_change = np.expand_dims(fft_phase_change,axis=2) #axis=0                    
                frame_temp[:,:,j:j+1] = fft_result_change
                frame_temp1[:,:,j:j+1] = fft_amplitude_change
                frame_temp2[:,:,j:j+1] = fft_phase_change
                                                                
            frame_z = np.append(frame_z,frame_temp,axis=0)  
            frame_abs = np.append(frame_abs ,frame_temp1,axis=0) 
            frame_phase = np.append(frame_phase ,frame_temp2,axis=0)
             
        print('frame_z = ',frame_z.shape)
            
        #做每個frame的phase差      
        frame_sub_z = np.zeros((0,64,15), dtype=complex)
        frame_sub_zphase = np.zeros((0,64,15), dtype=np.float32)
        frame_sub_zmag = np.zeros((0,64,15), dtype=np.float32)
        frame_allphase = np.zeros((0,64,15), dtype=np.float32)
               
        for i in range(frame_len-1):
            test = frame_z[i+1][:][:] - frame_z[i][:][:]
            test = np.expand_dims(test,axis=0)
            frame_sub_z = np.append(frame_sub_z,test,axis=0)
            frame_sub_zphase = np.append(frame_sub_zphase,np.angle(test),axis=0)
            frame_sub_zmag = np.append(frame_sub_zmag,np.abs(test),axis=0)

        #frame_allphase
        for i in range(frame_len-1):
            test = np.angle(frame_z[i+1][:][:]) - np.angle(frame_z[i][:][:])
            test = np.expand_dims(test,axis=0)
            frame_allphase = np.append(frame_allphase,test,axis=0)             
        
        #參數
        range_data = 6 
        frame_num=100
        #FFT的頻率軸
        frequencies1 = np.fft.fftfreq(padded_length,1/64)/32 #x
        frequencies_change1 = np.degrees(np.arcsin(np.concatenate((frequencies1[middle_index:], frequencies1[:middle_index]))))    
        
        #頻譜差值圖
        plt.figure(1)
        plt.plot(frequencies_change1,np.squeeze(frame_sub_zmag[frame_num:frame_num+1,:,range_data:range_data+1],axis=(2,0)))
        plt.xlabel('angle')
        plt.ylabel('mag')
        plt.title('rangeAzimuthHeatMap')        
                
        #找到angle最大值              
        angle_data = np.squeeze(frame_sub_zmag[frame_num:frame_num+1,:,range_data:range_data+1],axis=(2,0))
        angle_temp = 0
        for i in range(len(angle_data)):
            if angle_data[i]>angle_temp :
                angle_temp = angle_data[i]
                angle_max = i      
        print('angle_max = ',angle_max)
        #找到angle最大值的相位 
        temp = np.unwrap(np.squeeze(frame_phase[:200,angle_max:angle_max+1,range_data:range_data+1],axis=(2, 1))) #, markersize=8
        plt.figure(2)
        plt.plot(temp)
        #np.save('new_one_data.npz', data=temp)        
          
        '''
        #第一次存檔 
        data_mid = np.expand_dims(temp,axis=0)
        np.savez('../machine_learning/LSTM_Autoencoder/data/new_one_data.npz',data=data_mid)
        print(data_mid.shape)
        '''
        #先讀檔


        '''
        dataset = np.load('../machine_learning/LSTM_Autoencoder/data/new_one_data.npz') #vir_data_1Hz.npz
        load_mid = dataset['data']
        print(load_mid.shape)   
        '''
        '''
        #刪1個檔
        last_index = len(load_mid) - 1
        load_mid = np.delete(load_mid, last_index, axis=0)
        np.savez('../machine_learning/LSTM_Autoencoder/data/new_one_data.npz',mid=load_mid)
        print(load_mid.shape)
        '''
        '''
        #再存檔       

        
        temp = np.expand_dims(temp,axis=0) #升維
        data_mid = np.append(load_mid,temp,axis=0)
        np.savez('../machine_learning/LSTM_Autoencoder/data/new_one_data.npz',data=data_mid)
        print(data_mid.shape) 
        '''
        
        

        

        
        '''
        temp5 = np.expand_dims(load_mid[len(load_mid)-1],axis=0) #升維
        dataset_error = np.load('vir_data_freq_error.npz')
        load_error_mid = dataset_error['mid']
        data_error_mid = np.append(load_error_mid,temp5,axis=0)
        np.savez('vir_data_freq_error.npz',mid=data_error_mid)
        print(load_error_mid.shape)
        '''


                                              
        #R-Theta  
        x = np.linspace(-32, 32, 64) #Theta
        y = np.linspace(2.25, 65.25, 15) #R    
        # 生成熱圖
        # 計算頻率軸  #frequencies2 = frequencies1  frequencies_change2=frequencies_change1
        frequencies2 = np.fft.fftfreq(padded_length,1/64)/32 
        frequencies_change2 = np.degrees(np.arcsin(np.fft.fftshift(frequencies2)))        
        
        yi,xi= np.meshgrid(y,frequencies_change2[1:64]) 
        #頻譜差值熱圖
        plt.figure(3)
        plt.pcolormesh(xi,yi,frame_sub_zmag[frame_num,1:64,:],cmap='jet') 
        plt.colorbar()
        plt.xlabel('angle(\u03b8°)',fontsize=12)
        plt.ylabel('distnace(cm)',fontsize=12)
        plt.title('heat map',fontsize=12)
        #頻譜熱圖
        plt.figure(4)
        plt.pcolormesh(xi,yi,frame_abs[frame_num,1:64,:],cmap='jet') #, color='blue'  #,shading='auto' , vmin=-1, vmax=1
        plt.colorbar()
        plt.xlabel('angle(\u03b8°)')
        plt.ylabel('distnace(cm)')
        plt.title('heat map')  #rangeAzimuthHeatMap
                

        # 進行FFT 確認frame_phase是否在2Hz
        # 整個z值做FFT
        frame_Hz = np.fft.fft(np.squeeze(frame_z[50:200,angle_max:angle_max+1,range_data:range_data+1],axis=(2,1))) #y
        
        # 取得單邊頻譜  
        frequencies3 = np.fft.fftfreq(len(frame_Hz), delta_t) #x
        N = len(frame_Hz)
        positive_frequencies = frequencies3[:N//2]
        #  
        positive_fft = 2/N * np.abs(frame_Hz[:N//2]) #2/N比例因子
        positive_fft[0]=0
        
        # 繪製單邊頻譜圖
        plt.figure(5)
        plt.plot(positive_frequencies, positive_fft)
        plt.xlabel('Frequency[Hz]')
        plt.ylabel('Amplitude')            
        
        # z值的相位做FFT
        frame_Hz_phase = np.fft.fft(np.squeeze(frame_phase[50:200,angle_max:angle_max+1,range_data:range_data+1],axis=(2,1))) #y
        # 取得單邊頻譜
        positive_fft_phase = 2/N * np.abs(frame_Hz_phase[:N//2]) #2/N比例因子
        positive_fft_phase[0]=0
        # 繪製單邊頻譜圖
        plt.figure(6)
        plt.plot(positive_frequencies, positive_fft_phase)
        plt.xlabel('Frequency[Hz]')
        plt.ylabel('Amplitude')  
                                               
               
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        #win.close()
        break
      
