#include <Wire.h>
#include <EEPROM.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_L3GD20_U.h>

#define ACC_X_PIN 2
#define ACC_Y_PIN 1
#define ACC_Z_PIN 3

// Filtering variables
int trailingAccX1, trailingAccX2, trailingAccX3; // X3 - max X, X2 - min X
int trailingAccY1, trailingAccY2, trailingAccY3;
int trailingAccZ1, trailingAccZ2, trailingAccZ3;

int timesCalibrated = 0;
int timesCalibratedTotal = 0;
int timesCalibratedSinceOn = 0;

/* Assign a unique ID to this sensor at the same time */
Adafruit_L3GD20_Unified gyro = Adafruit_L3GD20_Unified(20);

void setup(void) 
{
  Serial.begin(115200);
  delay(1500);
  Serial.println("Gyroscope Test"); Serial.println("");
  
  /* Enable auto-ranging */
  gyro.enableAutoRange(true);
  
  /* Initialise the sensor */
  if(!gyro.begin())
  {
    /* There was a problem detecting the L3GD20 ... check your connections */
    Serial.println("Ooops, no L3GD20 detected ... Check your wiring!");
    
    while(1) {
      Serial.println("Ooops, no L3GD20 detected ... Check your wiring!");;
    }
  }
  
  /* Display some basic information on this sensor */
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

void loop(void) 
{
  /* Get a new sensor event */ 
  sensors_event_t event; 
  gyro.getEvent(&event);
 
  /* Display the results (speed is measured in rad/s) */
  //Serial.print("X: "); Serial.print(event.gyro.x); Serial.print("  ");
  //Serial.print("Y: "); Serial.print(event.gyro.y); Serial.print("  ");
  //Serial.print("Z: "); Serial.print(event.gyro.z); Serial.print("  ");
  //Serial.println("rad/s ");

  float acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;
  acc_x = filterAccX(analogRead(ACC_X_PIN));
  acc_y = filterAccY(analogRead(ACC_Y_PIN));
  acc_z = filterAccZ(analogRead(ACC_Z_PIN));
  
//  Serial.printf("S%f,%f,%f|%d,%d,%d\n", event.gyro.x, event.gyro.y, event.gyro.z, acc_x, acc_y, acc_z);
//  Serial.printf("S%f%f%f\n", event.gyro.x, event.gyro.y, event.gyro.z);
  
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
} 

void displaySensorDetails(void)
{
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

// Resets internal EEPROM memory saved settings
// and calibration data
void resetPrefs() {
  // X-min
  EEPROM.write(0, 0);
  EEPROM.write(0, 0);
  EEPROM.write(0, 0);
  EEPROM.write(0, 0);

  // TODO...
}

// Saves calibration data to EEPROM
void savePrefs(bool accX, bool accY, bool accZ) {
  // TODO: Serialize pref blocks to byte arrays and write to memory
  
  // Increment calibration counter  
  timesCalibrated += 1;
  timesCalibratedSinceOn += 1;
}
