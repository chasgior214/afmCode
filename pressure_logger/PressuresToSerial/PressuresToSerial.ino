// Analog input pins
const int pinA0 = A0;
const int pinA5 = A5;

// Sensor calibration constants
const float voltageZero = 0.5;   // Voltage at 0 PSI
const float voltageMax  = 4.5;   // Voltage at +150 PSI
const float psiRange    = 150.0;
const float kPaPerPsi   = 6.89476;

// Combined conversion factor
const float conversionFactor =
  (psiRange / (voltageMax - voltageZero)) * kPaPerPsi;

// Number of interleaved sample pairs per second
const int numPairs = 500; // 500 A0 reads + 500 A5 reads within ~1 second

static inline float adcToVoltage(int adc) {
  return adc * (5.0f / 1023.0f);
}

void setup() {
  Serial.begin(115200);  // higher baud to reduce print bottlenecks
}

void loop() {
  float sumV0 = 0.0f;
  float sumV5 = 0.0f;

  // Interleaved sampling window (~1 second total)
  // Each iteration reads both channels back-to-back.
  for (int i = 0; i < numPairs; i++) {
    int adc0 = analogRead(pinA0);
    int adc5 = analogRead(pinA5);

    sumV0 += adcToVoltage(adc0);
    sumV5 += adcToVoltage(adc5);

    delay(1000 / numPairs); // ~2 ms per pair when numPairs=500
  }

  float avgV0 = sumV0 / numPairs;
  float avgV5 = sumV5 / numPairs;

  float p0_kPa = (avgV0 - voltageZero) * conversionFactor;
  float p5_kPa = (avgV5 - voltageZero) * conversionFactor;

  // One machine-friendly line per second:
  // A0_V,A0_PkPa,A5_V,A5_PkPa
  Serial.print(avgV0, 3);
  Serial.print(",");
  Serial.print(p0_kPa, 2);
  Serial.print(",");
  Serial.print(avgV5, 3);
  Serial.print(",");
  Serial.println(p5_kPa, 2);
}
