#include <Adafruit_Sensor.h>
#include <Adafruit_L3GD20_U.h>
#include <AccelStepper.h>

#define ACC_X_PIN 2
#define ACC_Y_PIN 1
#define ACC_Z_PIN 3
#define DATA_COLLECTION_INTERVAL 20
#define DATA_BUFFER_SIZE 100

// Filtering variables
int trailingAccX1, trailingAccX2, trailingAccX3; // X3 - max X, X2 - min X
int trailingAccY1, trailingAccY2, trailingAccY3;
int trailingAccZ1, trailingAccZ2, trailingAccZ3;

Adafruit_L3GD20_Unified gyro = Adafruit_L3GD20_Unified(20);

#define M1_DIR 6
#define M1_STEP 7
#define M1_MICROSTEP 1
#define M2_DIR 11
#define M2_STEP 12
#define M2_MICROSTEP 1
#define M1_DISTANCE 100
#define M2_DISTANCE 100

AccelStepper m1(AccelStepper::DRIVER, M1_STEP, M1_DIR);
AccelStepper m2(AccelStepper::DRIVER, M2_STEP, M2_DIR);

void setup(void)  {
  Serial.begin(115200);
  delay(1500);

  setupAccelGyro();
  setupMotors();
}

void setupAccelGyro() {
  gyro.enableAutoRange(true);

  // Initialise the sensor
  if (!gyro.begin())
    Serial.println("Ooops, no L3GD20 detected ... Check your wiring!");
  else
    Serial.println("SPI Gyro found");

  // Display some basic information on this sensor
  displaySensorDetails();

  trailingAccX1 = analogRead(ACC_X_PIN);
  trailingAccX2 = analogRead(ACC_X_PIN);
  trailingAccX3 = analogRead(ACC_X_PIN);
  trailingAccY1 = analogRead(ACC_Y_PIN);
  trailingAccY2 = analogRead(ACC_Y_PIN);
  trailingAccY3 = analogRead(ACC_Y_PIN);
  trailingAccZ1 = analogRead(ACC_Z_PIN);
  trailingAccZ2 = analogRead(ACC_Z_PIN);
  trailingAccZ3 = analogRead(ACC_Z_PIN);
}

void setupMotors() {
  m1.setMaxSpeed(1000);
  m1.setAcceleration(100);
  m2.setMaxSpeed(1000);
  m2.setAcceleration(100);
}

// Filtering methods

float filterAccX(int curX) {
  if (curX < trailingAccX2) trailingAccX2 = curX;
  if (curX > trailingAccX3) trailingAccX3 = curX;

  return ((((float)(curX - trailingAccX2)) / (trailingAccX3 - trailingAccX2)) - 0.5f) * 2;
}

float filterAccY(int curY) {
  if (curY < trailingAccY2) trailingAccY2 = curY;
  if (curY > trailingAccY3) trailingAccY3 = curY;

  return ((((float)(curY - trailingAccY2)) / (trailingAccY3 - trailingAccY2)) - 0.5f) * 2;
}

float filterAccZ(int curZ) {
  if (curZ < trailingAccZ2) trailingAccZ2 = curZ;
  if (curZ > trailingAccZ3) trailingAccZ3 = curZ;

  return ((((float)(curZ - trailingAccZ2)) / (trailingAccZ3 - trailingAccZ2)) - 0.5f) * 2;
}

bool moveFw = true;
bool motorPause = false;
unsigned long pauseStart = 0;

void loop(void)  {
  RunMotor();

  if (CollectGyroAndAccelData())
    SendGyroAndAccelData();
}

void RunMotor() {
  bool m1active = m1.run();
  bool m2active = m2.run();

  if (m1active || m2active) {
    motorPause = true;
    pauseStart = millis();
  } else {
    if (motorPause) {
      if (millis() - pauseStart >= 1000)
        motorPause = false;
    } else {
      if (moveFw) {
        m1.moveTo(M1_DISTANCE * M1_MICROSTEP);
        m2.moveTo(M2_DISTANCE * M2_MICROSTEP);
      } else {
        m1.moveTo(0);
        m2.moveTo(0);
      }
      moveFw = !moveFw;
    }
  }
}

unsigned long lastCollectionTime = 0;
byte dataBuffer[DATA_BUFFER_SIZE][26];
unsigned int bufferIndex = 0;

bool CollectGyroAndAccelData() {
  if (millis() - lastCollectionTime < DATA_COLLECTION_INTERVAL)
    return false;

  sensors_event_t event;
  gyro.getEvent(&event);

  float acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;
  acc_x = filterAccX(analogRead(ACC_X_PIN));
  acc_y = filterAccY(analogRead(ACC_Y_PIN));
  acc_z = filterAccZ(analogRead(ACC_Z_PIN));
  //acc_x = analogRead(ACC_X_PIN);
  //acc_y = analogRead(ACC_Y_PIN);
  //acc_z = analogRead(ACC_Z_PIN);

  byte data[26] = { 'D',
                    // Gyro
                    ((byte*) &event.gyro.x)[0],
                    ((byte*) &event.gyro.x)[1],
                    ((byte*) &event.gyro.x)[2],
                    ((byte*) &event.gyro.x)[3],

                    ((byte*) &event.gyro.y)[0],
                    ((byte*) &event.gyro.y)[1],
                    ((byte*) &event.gyro.y)[2],
                    ((byte*) &event.gyro.y)[3],

                    ((byte*) &event.gyro.z)[0],
                    ((byte*) &event.gyro.z)[1],
                    ((byte*) &event.gyro.z)[2],
                    ((byte*) &event.gyro.z)[3],

                    // Accelerometer
                    ((byte*) &acc_x)[0],
                    ((byte*) &acc_x)[1],
                    ((byte*) &acc_x)[2],
                    ((byte*) &acc_x)[3],

                    ((byte*) &acc_y)[0],
                    ((byte*) &acc_y)[1],
                    ((byte*) &acc_y)[2],
                    ((byte*) &acc_y)[3],

                    ((byte*) &acc_z)[0],
                    ((byte*) &acc_z)[1],
                    ((byte*) &acc_z)[2],
                    ((byte*) &acc_z)[3],

                    '\n'
                  };

  if (bufferIndex == DATA_BUFFER_SIZE)
    bufferIndex = 0;
  memcpy(dataBuffer[bufferIndex], data, 26);
  bufferIndex++;

  lastCollectionTime = millis();

  return (bufferIndex == DATA_BUFFER_SIZE);
}

void SendGyroAndAccelData() {
  for (int i = 0; i < DATA_BUFFER_SIZE; i++)
    Serial.write(dataBuffer[i], 26);
}

/*void SendGyroAndAccelData() {
  // Get a new sensor event
  sensors_event_t event;
  gyro.getEvent(&event);

  float acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;
  acc_x = filterAccX(analogRead(ACC_X_PIN));
  acc_y = filterAccY(analogRead(ACC_Y_PIN));
  acc_z = filterAccZ(analogRead(ACC_Z_PIN));

  byte buf[26] = { 'D',
                   // Gyro
                   ((byte*) &event.gyro.x)[0],
                   ((byte*) &event.gyro.x)[1],
                   ((byte*) &event.gyro.x)[2],
                   ((byte*) &event.gyro.x)[3],

                   ((byte*) &event.gyro.y)[0],
                   ((byte*) &event.gyro.y)[1],
                   ((byte*) &event.gyro.y)[2],
                   ((byte*) &event.gyro.y)[3],

                   ((byte*) &event.gyro.z)[0],
                   ((byte*) &event.gyro.z)[1],
                   ((byte*) &event.gyro.z)[2],
                   ((byte*) &event.gyro.z)[3],

                   // Accelerometer
                   ((byte*) &acc_x)[0],
                   ((byte*) &acc_x)[1],
                   ((byte*) &acc_x)[2],
                   ((byte*) &acc_x)[3],

                   ((byte*) &acc_y)[0],
                   ((byte*) &acc_y)[1],
                   ((byte*) &acc_y)[2],
                   ((byte*) &acc_y)[3],

                   ((byte*) &acc_z)[0],
                   ((byte*) &acc_z)[1],
                   ((byte*) &acc_z)[2],
                   ((byte*) &acc_z)[3],

                   '\n'  };

  Serial.write(buf, 26);
  delay(20);
  }*/

void displaySensorDetails(void) {
  sensor_t sensor;
  gyro.getSensor(&sensor);
  Serial.println("------------------------------------");
  Serial.print  (" Sensor:       "); Serial.println(sensor.name);
  Serial.print  (" Driver Ver:   "); Serial.println(sensor.version);
  Serial.print  (" Unique ID:    "); Serial.println(sensor.sensor_id);
  Serial.print  (" Max Value:    "); Serial.print(sensor.max_value); Serial.println(" rad/s");
  Serial.print  (" Min Value:    "); Serial.print(sensor.min_value); Serial.println(" rad/s");
  Serial.print  (" Resolution:   "); Serial.print(sensor.resolution); Serial.println(" rad/s");
  Serial.println("------------------------------------");
  Serial.println("");
  delay(100);
}
