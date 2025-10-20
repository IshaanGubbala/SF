import time
from gpiozero import Button
from signal import pause
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from luma.core.render import canvas
from PIL import ImageFont
import spidev

# -------------------------
# Configuration & Constants
# -------------------------
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 64
I2C_ADDRESS = 0x3C  # OLED I2C address

# GPIO Pins for Encoder & Button
ENC_CLK_PIN = 17   # Encoder clock pin
ENC_DT_PIN  = 27   # Encoder data pin
ENC_SW_PIN  = 22   # Encoder push-button

# ADC channels for sensors
VOC_CHANNEL = 0    # VOC sensor channel
FEV1_CHANNEL = 1   # FEV1 sensor channel

STANDARD_PPB = 550

# Menu states
M_HEIGHT, M_GENDER, M_AGE, M_TEST, M_RESULTS = range(5)
currentMenu = M_HEIGHT

# Global variables for settings and sensor values
height = 60
isMale = True
age = 30
testing = False
offsetFEV1 = 1.2
offsetVOC = 0
time_start = 0
baselineReading = 0
maxVOC = 0
maxFEV1 = 0
currentVOC = 0
currentFEV1 = 0

# -------------------------
# SPI/ADC Setup via spidev (same as before)
# -------------------------
spi = spidev.SpiDev()
spi.open(0, 0)  # Open bus 0, device 0
spi.max_speed_hz = 1350000

def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

def convertToVoltage(adc_value):
    return (adc_value * 5.0) / 1023.0

def convertToPPM(voltage):
    return voltage * 2.0  # Adjust calibration as needed

def constrain(val, low, high):
    return min(max(val, low), high)

# -------------------------
# OLED Display Setup (unchanged)
# -------------------------
serial = i2c(port=1, address=I2C_ADDRESS)
device = sh1106(serial, rotate=0)

try:
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception as e:
    font_small = ImageFont.load_default()
    font_large = ImageFont.load_default()

def updateDisplay():
    predFEV1 = (0.0414 * height * 2.54) - (0.0244 * age) - 2.190
    with canvas(device) as draw:
        if currentMenu == M_HEIGHT:
            draw.text((0, 0), "Set Height:", fill="white", font=font_small)
            draw.text((0, 15), f"{height} in", fill="white", font=font_large)
        elif currentMenu == M_GENDER:
            draw.text((0, 0), "Set Gender:", fill="white", font=font_small)
            gender_text = "Male" if isMale else "Female"
            draw.text((0, 15), gender_text, fill="white", font=font_large)
        elif currentMenu == M_AGE:
            draw.text((0, 0), "Set Age:", fill="white", font=font_small)
            draw.text((0, 15), f"{age} yr", fill="white", font=font_large)
        elif currentMenu == M_TEST:
            if testing:
                fev1_val = convertToPPM(convertToVoltage(currentFEV1)) - offsetFEV1
                voc_val = convertToPPM(convertToVoltage(currentVOC)) - offsetVOC
                risk = (convertToVoltage(currentVOC) > 0.24) and (predFEV1 * 0.8 > (convertToPPM(convertToVoltage(currentFEV1)) - offsetFEV1))
                draw.text((0, 0), f"FEV1: {fev1_val:.2f}", fill="white", font=font_small)
                draw.text((0, 12), f"VOC: {voc_val:.2f}", fill="white", font=font_small)
                risk_text = "Risk: Yes" if risk else "Risk: No"
                draw.text((0, 24), risk_text, fill="white", font=font_small)
            else:
                draw.text((0, 0), "Ready to Test", fill="white", font=font_small)
                draw.text((0, 12), "Press button", fill="white", font=font_small)
                draw.text((0, 24), "to start", fill="white", font=font_small)
        elif currentMenu == M_RESULTS:
            avgFEV1 = currentFEV1  # Placeholder for average FEV1
            fev1_thresh = (convertToPPM(convertToVoltage(avgFEV1)) * 100 / predFEV1) if predFEV1 != 0 else 0
            draw.text((0, 0), "Results:", fill="white", font=font_small)
            draw.text((0, 12), f"Max VOC: {maxVOC:.2f}", fill="white", font=font_small)
            draw.text((0, 24), f"FEV1 Thresh: {fev1_thresh:.2f}%", fill="white", font=font_small)
            risk_text = "Risk Detected" if (convertToVoltage(currentVOC) > 0.24 and predFEV1 * 0.8 > (convertToPPM(convertToVoltage(currentFEV1))-offsetFEV1)) else "No Risk"
            draw.text((0, 36), risk_text, fill="white", font=font_small)

def calibrateVOC():
    vocSum = 0.0
    numReadings = 10
    for i in range(numReadings):
        vocValue = read_adc(VOC_CHANNEL)
        vocVoltage = vocValue * (5.0 / 1023.0)
        Rs = (5.0 - vocVoltage) * 100 / vocVoltage if vocVoltage != 0 else 0
        ratio = Rs / 100
        try:
            ppb = 1000.0 / (ratio - 0.2)
        except ZeroDivisionError:
            ppb = 0
        vocSum += ppb
        time.sleep(0.1)
    return (vocSum / numReadings) - STANDARD_PPB

baselineReading = calibrateVOC()
updateDisplay()

# -------------------------
# Using gpiozero for Input Handling
# -------------------------
from gpiozero import Button

# Create Button objects for the encoder and push switch
encoder_clk = Button(ENC_CLK_PIN, pull_up=True)
encoder_dt  = Button(ENC_DT_PIN, pull_up=True)
encoder_sw  = Button(ENC_SW_PIN, pull_up=True)

# Callback for the rotary encoder
def on_encoder_change():
    global height, isMale, age, currentMenu
    # Determine rotation direction based on the state of encoder_dt
    if encoder_dt.is_pressed:
        # Clockwise: increment value
        if currentMenu == M_HEIGHT:
            height = constrain(height + 1, 48, 84)
        elif currentMenu == M_AGE:
            age = constrain(age + 1, 5, 100)
        elif currentMenu == M_GENDER:
            # Toggle gender for simplicity
            global isMale
            isMale = not isMale
    else:
        # Counter-clockwise: decrement value
        if currentMenu == M_HEIGHT:
            height = constrain(height - 1, 48, 84)
        elif currentMenu == M_AGE:
            age = constrain(age - 1, 5, 100)
        elif currentMenu == M_GENDER:
            isMale = not isMale
    updateDisplay()

# Set up the encoder clock to call our handler when pressed (falling edge)
encoder_clk.when_pressed = on_encoder_change

# Callback for the push button
def on_button_press():
    global currentMenu, testing, time_start, maxVOC, maxFEV1
    # Simple debouncing is handled by gpiozero
    if currentMenu == M_TEST:
        if not testing:
            testing = True
            maxVOC = 0
            maxFEV1 = 0
            time_start = time.time() * 1000  # in ms
        else:
            testing = False
            currentMenu = (currentMenu + 1) % 5
    else:
        currentMenu = (currentMenu + 1) % 5
    updateDisplay()

encoder_sw.when_pressed = on_button_press

# -------------------------
# Main Loop for Sensor Reading
# -------------------------
lastSensorUpdate = 0
try:
    while True:
        if testing:
            current_time = time.time() * 1000  # milliseconds
            if (current_time - lastSensorUpdate) > 100:
                currentVOC = read_adc(VOC_CHANNEL) - baselineReading
                currentFEV1 = read_adc(FEV1_CHANNEL)
                lastSensorUpdate = current_time
                updateDisplay()
        time.sleep(0.01)
except KeyboardInterrupt:
    spi.close()
