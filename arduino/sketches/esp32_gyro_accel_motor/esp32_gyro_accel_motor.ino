#include <driver/i2c.h>
#include <esp_log.h>
#include <esp_err.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <MPU6050.h>
#include <MPU6050_6Axis_MotionApps20.h>
#include <AccelStepper.h>

#define PIN_SDA 22
#define PIN_CLK 21

#define DATA_COLLECTION_INTERVAL 20
#define DATA_BUFFER_SIZE 100


// Filtering variables
int trailingAccX1, trailingAccX2, trailingAccX3; // X3 - max X, X2 - min X
int trailingAccY1, trailingAccY2, trailingAccY3;
int trailingAccZ1, trailingAccZ2, trailingAccZ3;

#define M1_DIR 6
#define M1_STEP 7
#define M1_MICROSTEP 1
#define M2_DIR 11
#define M2_STEP 12
#define M2_MICROSTEP 1
#define M1_DISTANCE 100
#define M2_DISTANCE 100

//AccelStepper m1(AccelStepper::DRIVER, M1_STEP, M1_DIR);
//AccelStepper m2(AccelStepper::DRIVER, M2_STEP, M2_DIR);

Quaternion q;           // [w, x, y, z]         quaternion container
VectorFloat gravity;    // [x, y, z]            gravity vector
float ypr[3];           // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector
uint16_t packetSize = 42;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer
uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU

MPU6050 mpu = MPU6050();

bool runOnce = false;

void setup(void) {
    Serial.begin(115200);
    delay(500);

    esp32i2cInit();

    delay(500);

    Serial.println("Initializing...");

    setupAccelGyro();
    //setupMotors();
}

void esp32i2cInit() {
    i2c_config_t conf;
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = (gpio_num_t) PIN_SDA;
    conf.scl_io_num = (gpio_num_t) PIN_CLK;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = 400000;
    ESP_ERROR_CHECK(i2c_param_config(I2C_NUM_0, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_NUM_0, I2C_MODE_MASTER, 0, 0, 0));
}

void setupAccelGyro() {
    mpu.initialize();
    Serial.println("mpu initialize...");
    mpu.dmpInitialize();
    Serial.println("mpu dmpInitialize...");

    // This need to be setup individually
    mpu.setXGyroOffset(220);
    mpu.setYGyroOffset(76);
    mpu.setZGyroOffset(-85);
    mpu.setZAccelOffset(1788);

    mpu.setDMPEnabled(true);
    Serial.println("mpu dmpEnabled...");

    Serial.println("MPU6050 ready...");

    // Display some basic information on this sensor
    displaySensorDetails();

    // Pre-fill 'trailing' values for use in filtering
//  trailingAccX1 = analogRead(ACC_X_PIN);
//  trailingAccX2 = analogRead(ACC_X_PIN);
//  trailingAccX3 = analogRead(ACC_X_PIN);
//  trailingAccY1 = analogRead(ACC_Y_PIN);
//  trailingAccY2 = analogRead(ACC_Y_PIN);
//  trailingAccY3 = analogRead(ACC_Y_PIN);
//  trailingAccZ1 = analogRead(ACC_Z_PIN);
//  trailingAccZ2 = analogRead(ACC_Z_PIN);
//  trailingAccZ3 = analogRead(ACC_Z_PIN);
}

void setupMotors() {
//    m1.setMaxSpeed(1000);
//    m1.setAcceleration(100);
//    m2.setMaxSpeed(1000);
//    m2.setAcceleration(100);
}

// Filtering methods

float filterAccX(int curX) {
    if (curX < trailingAccX2) trailingAccX2 = curX;
    if (curX > trailingAccX3) trailingAccX3 = curX;

    return ((((float) (curX - trailingAccX2)) / (trailingAccX3 - trailingAccX2)) - 0.5f) * 2;
}

float filterAccY(int curY) {
    if (curY < trailingAccY2) trailingAccY2 = curY;
    if (curY > trailingAccY3) trailingAccY3 = curY;

    return ((((float) (curY - trailingAccY2)) / (trailingAccY3 - trailingAccY2)) - 0.5f) * 2;
}

float filterAccZ(int curZ) {
    if (curZ < trailingAccZ2) trailingAccZ2 = curZ;
    if (curZ > trailingAccZ3) trailingAccZ3 = curZ;

    return ((((float) (curZ - trailingAccZ2)) / (trailingAccZ3 - trailingAccZ2)) - 0.5f) * 2;
}

bool moveFw = true;
bool motorPause = false;
unsigned long pauseStart = 0;

void loop(void) {
    if (!runOnce) {
        CollectGyroAndAccelData();
        runOnce = true;
    }
    //Serial.println("test");
    //delay(1000);
    //RunMotor();

    if (CollectGyroAndAccelData()) {
        SendGyroAndAccelData();
    }
}

void RunMotor() {
//    bool m1active = m1.run();
//    bool m2active = m2.run();
//
//    if (m1active || m2active) {
//        motorPause = true;
//        pauseStart = millis();
//    } else {
//        if (motorPause) {
//            if (millis() - pauseStart >= 1000)
//                motorPause = false;
//        } else {
//            if (moveFw) {
//                m1.moveTo(M1_DISTANCE * M1_MICROSTEP);
//                m2.moveTo(M2_DISTANCE * M2_MICROSTEP);
//            } else {
//                m1.moveTo(0);
//                m2.moveTo(0);
//            }
//            moveFw = !moveFw;
//        }
//    }
}

unsigned long lastCollectionTime = 0;
byte dataBuffer[DATA_BUFFER_SIZE][26];
unsigned int bufferIndex = 0;

bool CollectGyroAndAccelData() {
    if (millis() - lastCollectionTime < DATA_COLLECTION_INTERVAL)
        return false;

    mpuIntStatus = mpu.getIntStatus();
    // get current FIFO count
    fifoCount = mpu.getFIFOCount();

    if ((mpuIntStatus & 0x10) || fifoCount == 1024) {
        // reset so we can continue cleanly
        mpu.resetFIFO();

        // otherwise, check for DMP data ready interrupt frequently)
    } else if (mpuIntStatus & 0x02) {
        // wait for correct available data length, should be a VERY short wait
        while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();

        // read a packet from FIFO

        mpu.getFIFOBytes(fifoBuffer, packetSize);
        mpu.dmpGetQuaternion(&q, fifoBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
        Serial.print("X: ");
        Serial.print(gravity.x);
        Serial.print(" Y: ");
        Serial.print(gravity.y);
        Serial.print(" Z: ");
        Serial.print(gravity.z);
        Serial.print(" | YAW: ");
        Serial.print(ypr[0] * 180 / M_PI);
        Serial.print(" PITCH: ");
        Serial.print(ypr[1] * 180 / M_PI);
        Serial.print(" ROLL: ");
        Serial.println(ypr[2] * 180 / M_PI);
    }

//  float acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;
//  acc_x = filterAccX(analogRead(ACC_X_PIN));
//  acc_y = filterAccY(analogRead(ACC_Y_PIN));
//  acc_z = filterAccZ(analogRead(ACC_Z_PIN));

    // Disabled buffered output
//  byte data[26] = { 'D',
//                    // Gyro
//                    ((byte*) &event.gyro.x)[0],
//                    ((byte*) &event.gyro.x)[1],
//                    ((byte*) &event.gyro.x)[2],
//                    ((byte*) &event.gyro.x)[3],
//
//                    ((byte*) &event.gyro.y)[0],
//                    ((byte*) &event.gyro.y)[1],
//                    ((byte*) &event.gyro.y)[2],
//                    ((byte*) &event.gyro.y)[3],
//
//                    ((byte*) &event.gyro.z)[0],
//                    ((byte*) &event.gyro.z)[1],
//                    ((byte*) &event.gyro.z)[2],
//                    ((byte*) &event.gyro.z)[3],
//
//                    // Accelerometer
//                    ((byte*) &acc_x)[0],
//                    ((byte*) &acc_x)[1],
//                    ((byte*) &acc_x)[2],
//                    ((byte*) &acc_x)[3],
//
//                    ((byte*) &acc_y)[0],
//                    ((byte*) &acc_y)[1],
//                    ((byte*) &acc_y)[2],
//                    ((byte*) &acc_y)[3],
//
//                    ((byte*) &acc_z)[0],
//                    ((byte*) &acc_z)[1],
//                    ((byte*) &acc_z)[2],
//                    ((byte*) &acc_z)[3],
//
//                    '\n'
//                  };
//
//  if (bufferIndex == DATA_BUFFER_SIZE)
//    bufferIndex = 0;
//  memcpy(dataBuffer[bufferIndex], data, 26);
//  bufferIndex++;

    lastCollectionTime = millis();

    return (bufferIndex == DATA_BUFFER_SIZE);
}

void SendGyroAndAccelData() {
    for (int i = 0; i < DATA_BUFFER_SIZE; i++)
        Serial.write(dataBuffer[i], 26);
}

void displaySensorDetails(void) {
//  sensor_t sensor;
//  gyro.getSensor(&sensor);
//  Serial.println("------------------------------------");
//  Serial.print  (" Sensor:       "); Serial.println(sensor.name);
//  Serial.print  (" Driver Ver:   "); Serial.println(sensor.version);
//  Serial.print  (" Unique ID:    "); Serial.println(sensor.sensor_id);
//  Serial.print  (" Max Value:    "); Serial.print(sensor.max_value); Serial.println(" rad/s");
//  Serial.print  (" Min Value:    "); Serial.print(sensor.min_value); Serial.println(" rad/s");
//  Serial.print  (" Resolution:   "); Serial.print(sensor.resolution); Serial.println(" rad/s");
    Serial.println("------------------------------------");
    Serial.println("");
    delay(100);
}
