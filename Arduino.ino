#include <Servo.h>

Servo myservo;  // Create servo object
const int trigPin = 9;  // Ultrasonic sensor TRIG pin
const int echoPin = 10; // Ultrasonic sensor ECHO pin
int angle = 75;         // Starting angle
int direction = 1;      // 1 for increasing angle, -1 for decreasing

void setup() {
  Serial.begin(9600);   // Start serial communication
  myservo.attach(11);   // Attach servo on pin 11
  
  // Setup ultrasonic sensor pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  // Start position
  myservo.write(angle);
  delay(1000);
}

void loop() {
  // Move servo
  angle += direction;
  
  // Change direction at limits
  if (angle >= 105) {
    direction = -1;
  }
  if (angle <= 75) {
    direction = 1;
  }
  
  // Move servo to new position
  myservo.write(angle);
  
  // Get distance measurement
  long duration, distance;
  
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;  // Convert to centimeters
  
  // Send data to Python
  Serial.print("Angle:");
  Serial.print(angle);
  Serial.print(",Distance:");
  Serial.print(distance);
  Serial.println("cm");
  
  delay(100);  // Small delay for stability
}
