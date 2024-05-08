import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
import tensorflow as tf

# 设置数据路径
train_data_dir = '/home/xzhang/work/subjects/zgj1/MAE_ViT/dataset_blurred_max/train'
val_data_dir = '/home/xzhang/work/subjects/zgj1/MAE_ViT/dataset_blurred_max/val'
test_data_dir = '/home/xzhang/work/subjects/zgj1/MAE_ViT/dataset_blurred_max/test'

# 设置图片大小和批量大小
img_width, img_height = 224, 224
batch_size = 16

# 创建 ImageDataGenerator 实例

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 使用 flow_from_directory 方法加载数据
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
#print(train_generator.classes)
# 创建基础模型
base_model = ResNet50(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             TruePositives(name='tp'),
             TrueNegatives(name='tn'),
             FalsePositives(name='fp'),
             FalseNegatives(name='fn')],
)

# 设置回调
checkpoint = ModelCheckpoint(
    filepath='/home/xzhang/work/subjects/zgj1/res_CNN/modelre-{epoch:02d}.hdf5',
    save_weights_only=False,
    save_best_only=True,
    monitor='val_auc',
    mode='max',
    verbose=1
)

# 训练模型
epochs = 100
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[checkpoint]
)
