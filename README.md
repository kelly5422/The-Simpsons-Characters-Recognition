# classification報告
## Machine Learning@NTUT - Classification報告
The Simpsons Characters Recognition Challenge

- 學生: 郭靜
- 學號: 108598068

---

## 做法說明
1. 將訓練的圖片讀進來,並根據資料夾的名稱製作標籤
2. 將訓練集的圖片90%用來訓練,10%拿來當驗證集
3. 定義模型
4. 將圖片檔轉為可以訓練的資料
5. 開始訓練模型
6. 將測試的圖片讀進來
7. 開始測試模型
8. 輸出csv檔

---

## 程式方塊圖與寫法

![](https://i.imgur.com/owt7PdT.png)


#### 將訓練的圖片讀進來,並根據資料夾的名稱製作標籤
```
# Read training images
images, labels = read_images_labels('/home/lab1323/Desktop/ml_classifivation_homework2/train/characters-20')

def read_images_labels(path, i):
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))
        if os.path.isdir(abs_path):
            i += 1
            temp = os.path.split(abs_path)[-1]
            name.append(temp)
            read_images_labels(abs_path,i)
            amount = int(len(os.listdir(path)))
            sys.stdout.write('\r'+'>'*(i)+' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp)
        else:
            if file.endswith('.jpg'):
                image = cv2.resize(cv2.imread(abs_path),(64,64))
                images.append(image)
                labels.append(i-1)
    return images, labels, name
```


#### 將訓練集的圖片90%用來訓練,10%拿來當驗證集
```
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
```
#### 定義模型
```
model = Sequential()
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
```
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_13 (Conv2D)           (None, 64, 64, 64)        1792      
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 32, 32, 64)        0         
_________________________________________________________________
dropout_9 (Dropout)          (None, 32, 32, 64)        0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 32, 32, 128)       73856     
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 32, 32, 128)       147584    
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 16, 16, 128)       0         
_________________________________________________________________
dropout_10 (Dropout)         (None, 16, 16, 128)       0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 16, 16, 256)       295168    
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 16, 16, 256)       590080    
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 16, 16, 256)       590080    
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 8, 8, 256)         0         
_________________________________________________________________
dropout_11 (Dropout)         (None, 8, 8, 256)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 16384)             0         
_________________________________________________________________
dense_5 (Dense)              (None, 512)               8389120   
_________________________________________________________________
dropout_12 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 20)                10260     
=================================================================
Total params: 10,097,940
Trainable params: 10,097,940
Non-trainable params: 0

```

#### 將圖片檔轉為可以訓練的資料
```
# from jpeg to data
datagen = ImageDataGenerator(zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train)
```

#### 開始訓練模型
```
epochs = 100
batch_size = 64
file_name = str(epochs) + '_' + str(batch_size)

# training
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_data=(X_test, y_test))
```

#### 將測試的圖片讀進來
```
test_images = read_images('test/test/')

def read_images(path):
    images=[]
    for i in range(990):
        image = cv2.resize(cv2.imread(path+str(i+1)+'.jpg'), (64,64))
        images.append(image)
    images = np.array(images,dtype=np.float32)/255
    return images
```

#### 開始測試模型
```
predict = model.predict_classes(test_images, verbose=1)
```

#### 輸出csv檔
```
raw_data={'id':range(1,991),
          'character':label_str    
}
df=pd.DataFrame(raw_data,columns=['id','character'])
df.to_csv('predict.csv',index=False,float_format='%.0f')
```

---

## 畫圖結果分析
* 下圖為訓練結果較不好的分析圖
epochs = 100
batch_size = 64
![](https://i.imgur.com/d5fSk1f.png)
![](https://i.imgur.com/45vp8VD.png)



* 下圖為訓練結果較好的分析圖
fitting程度較上圖好
epochs = 100
batch_size = 256
![](https://i.imgur.com/MUphAVB.png)
![](https://i.imgur.com/mrOhWXq.png)





---

## 討論預測值誤差很大的，是怎麼回事？
1. 當batch_size設定較小時,像是64,訓練效果並不好,可能是因為值太小導致機器能學習到的資訊不夠多,所以才造成訓練效果不好
2. epochs太大,對模型來說並沒有明顯的提昇

---

## 如何改進？
1. 最後我將batch_size設定在256,效果明顯比64時好許多
2. 我大概設定在100上下,發現設的太高或太低,對模型都沒有幫助

---# The-Simpsons-Characters-Recognition
