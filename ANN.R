### Deep Learning with Keras in R
### Date: 20180311

### Load in the library
library(keras)

### Load in the dataset of handwritten 0-9
mnist <- dataset_mnist()

### Preprocessing
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
# reshape
x_train <- array_reshape(x_train, dim=c(nrow(x_train), 28*28))
x_test <- array_reshape(x_test, dim=c(nrow(x_test), 28*28))
# rescale: normalization
x_train <- x_train / 255
x_test <- x_test / 255
# categorize repond
y_train <- to_categorical(y_train, num_classes=10)
y_test <- to_categorical(y_test, num_classes=10)

### Define the model: sequential model (building NN)
model <- keras_model_sequential()
model %>% 
  layer_dense(units=256, activation='relu', input_shape=c(28*28)) %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units=128, activation='relu') %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units=10, activation='softmax')

### Compile the model (set NN ready for training)
model %>% compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=c('accuracy')
)

### Train the model
history <- model %>% fit(
  x=x_train, y=y_train,
  batch_size=128, epochs=30,
  validation_split=0.2
)
#The model performs best at around 10 epochs, 
#then starts overfitting
plot(history)

### Predict and Evaluate the model
#predict
pred <- model %>% predict_classes(x_test) %>% as.factor()
#evaluate by hand
ans <- factor(apply(y_test,1,which.max)-1)
table(pred, ans)
accuracy <- sum(pred == ans)/nrow(x_test)
#built-in evaluate
model %>% evaluate(x_test, y_test)

### Tune the model


### Save the model
save_model_hdf5(model, 'firstNN.h5')
### Load the model
#model <- load_model_hdf5('firstNN.h5')

### Export the model in JSON or YAML
json_string <- model_to_json(model)
yaml_string <- model_to_yaml(model)

### Import the model from JSON or YAML
model <- model_from_json(json_string)
model <- model_from_yaml(yaml_string)

