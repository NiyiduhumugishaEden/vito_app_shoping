#include <SoftwareSerial.h>


SoftwareSerial configBt(10, 11); // RX, TX
void setup(){
  Serial.begin(9600);
  configBt.begin(9600);
  pinMode(13, OUTPUT); // Set pin 13 (built-in LED) as output
}


void loop(){

  if(configBt.available()){
    String receivedMsg = configBt.readString();
    Serial.print("Received message: ");
    Serial.println(receivedMsg);
    // Check the received message and take action
    if(receivedMsg.indexOf("on") != -1){
      digitalWrite(13, HIGH); // Turn LED on
      Serial.println("LED turned on");
    }
    else if(receivedMsg.indexOf("off") != -1){
      digitalWrite(13, LOW); // Turn LED off
      Serial.println("LED turned off");
    }
    // Add more conditions as needed based on your message format
    // Clear the received message buffer
    receivedMsg = "";
  }
  
  if(Serial.available()){
    configBt.write(Serial.read());
  }

}