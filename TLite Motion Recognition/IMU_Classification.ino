/*
  IMU Classifier for Arduino Nicla Vision
  
  This example uses the on-board BHY2 IMU to read acceleration and gyroscope data.
  When a significant motion is detected, it captures a window of data and uses a 
  TensorFlow Lite model to classify the gesture.

  This code is a corrected version for the Nicla Vision board.
*/

// CORRECT LIBRARY for Nicla Vision IMU
#include <Arduino_BHY2.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Your pre-trained model
#include "model.h"

// Globals for TensorFlow Lite
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Memory buffer for TFLM. Increase if you get "AllocateTensors() failed".
constexpr int tensorArenaSize = 8 * 1024;
alignas(16) uint8_t tensor_arena[tensorArenaSize];

// --- MODEL & GESTURE PARAMETERS (VERIFY THESE!) ---
// This must match the window size used during training
const int numSamples = 104; 

// The threshold to start capturing data (in G's)
const float accelerationThreshold = 2.5; 

// Array to map gesture index to a name (must match model output order)
const char* GESTURES[] = {
  "lateral", // Corresponds to model output[0]
  "updown"   // Corresponds to model output[1]
};
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// --- Global variables for data capture ---
int samplesRead = numSamples;
SensorXYZ accel(SENSOR_ID_ACC);
SensorXYZ gyro(SENSOR_ID_GYRO);

void setup() {
  Serial.begin(115200); // Using a faster baud rate is recommended
  while (!Serial);

  // Initialize the BHY2 sensor hub on the Nicla Vision
  BHY2.begin();
  
  // Configure the accelerometer and gyroscope
  // Note: The sample rate should ideally match the rate used during training
  if (!BHY2.configureSensor(accel, 100.0f, 0) || !BHY2.configureSensor(gyro, 100.0f, 0)) {
    Serial.println("Failed to configure IMU sensors!");
    while (1);
  }

  Serial.println("IMU Initialized. Ready for gestures.");
  Serial.println();

  // Load the model
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Set up the interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(tflModel, resolver, tensor_arena, tensorArenaSize);
  tflInterpreter = &static_interpreter;

  // Allocate memory for tensors
  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while(1);
  }

  // Get pointers to input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  // Update sensor data
  BHY2.update();

  // 1. Wait for a significant motion to trigger capture
  if (samplesRead == numSamples) {
    if (accel.hasNewData()) {
      float aSum = fabs(accel.x()) + fabs(accel.y()) + fabs(accel.z());
      if (aSum >= accelerationThreshold) {
        samplesRead = 0; // Reset counter and start capturing
        Serial.println("Motion detected, capturing data...");
      }
    }
    return; // Keep waiting if not triggered
  }

  // 2. Capture a full window of data
  if (samplesRead < numSamples) {
    if (accel.hasNewData() && gyro.hasNewData()) {
      // Get raw sensor values
      float aX = accel.x();
      float aY = accel.y();
      float aZ = accel.z();
      float gX = gyro.x();
      float gY = gyro.y();
      float gZ = gyro.z();

      // !! CRITICAL !!
      // This normalization must EXACTLY match the preprocessing from your training script.
      // Assuming Accel range is -4G to +4G
      // Assuming Gyro range is -2000dps to +2000dps
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;
      
      samplesRead++;

      // 3. If window is full, run inference
      if (samplesRead == numSamples) {
        Serial.println("Running inference...");
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          return;
        }

        // Print the results
        for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 6); // Print score with 6 decimal places
        }
        Serial.println("------------------------------------");
      }
    }
  }
}