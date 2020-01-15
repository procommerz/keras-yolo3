from __future__ import print_function
import threading
import os
import time
import sys
import RPi.GPIO as GPIO
import numpy
import json
import pdb

from flask import Flask
from flask import render_template
from flask import request

OUT_PIN = 23
IN_PIN = 25
TRIG_PIN = 23

IN2_PIN = 20

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(True)

app = None
meter = None
flaskapp = Flask(__name__)

class Meter(threading.Thread):
    def __init__(self):
        print("Initializing Meter")
        GPIO.setup(IN2_PIN, GPIO.IN)

        threading.Thread.__init__(self)
        self.deamon = True

        self.chart_data = list()
        self.chart_labels = list()

        self.countdown = 0
        self.start()

    def run(self):
        print("Running Meter")
        while 1 and app.main_thread.isAlive():
            # print(GPIO.input(IN2_PIN))
            self.chart_labels.append("%d" % (0.01 * self.countdown * 1000))
            self.chart_data.append(GPIO.input(IN2_PIN))
            self.countdown += 1
            time.sleep(0.01)

            # if self.countdown > 1000:
            #     pdb.set_trace()

class Emitter(threading.Thread):
    def __init__(self):
        print("Initializing Emitter")
        GPIO.setup(IN2_PIN, GPIO.IN)
        GPIO.setup(TRIG_PIN, GPIO.OUT)

        threading.Thread.__init__(self)

        self.countdown = 0

        self.start()

    def distance(self):
        GPIO.output(TRIG_PIN, True)
        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        while GPIO.input(IN_PIN) == 0:
            start_time = time.time()
        # save time of arrival
        while GPIO.input(IN_PIN) == 1:
            stop_time = time.time()
        # time difference between start and arrival
        elapsed = start_time - stop_time
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (elapsed * 34300) / 2
        return distance

    def emit_test(self, length):
        GPIO.output(TRIG_PIN, True)
        # set Trigger after 0.01ms to LOW
        time.sleep(length)
        GPIO.output(TRIG_PIN, False)

    def run(self):
        print("Running Emitter")
        while 1 and app.main_thread.isAlive():
            if app.enable_emitter:
                self.emit_test(0.00001)
            time.sleep(0.00001 * 400)

class SonarApp:
    def __init__(self, main_thread):
        self.main_thread = main_thread
        self.enable_emitter = True
        self.enable_meter = True
        pass

    def sonar(self):
        self.emitter = Emitter()
        self.meter = Meter()

    # def measure(self):
    #     GPIO.setup(TRIG_PIN, GPIO.OUT)
    #     GPIO.setup(IN_PIN, GPIO.IN)
    #
    #     try:
    #         while 1:
    #             res = self.distance()
    #             sys.stdout.flush()
    #             sys.stdout.write('Distance: %f cm\r' % round(res, 1))
    #             # print('Distance: %f cm\r' % round(res, 1), file=sys.stderr)
    #             time.sleep(0.01)
    #     except KeyboardInterrupt:
    #         pass
    #
    #     GPIO.cleanup()


    def listen(self):
        GPIO.setup(OUT_PIN, GPIO.OUT)
    # 	GPIO.setup(IN_PIN, GPIO.IN)
        GPIO.setup(IN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    # 	p1 = GPIO.PWM(OUT_PIN, 180)
    # 	p1.start(2)

        try:
            while 1:
    			# sys.stdout.write('%f\r' % GPIO.input(IN_PIN))
    			# sys.stdout.flush()
                reading = GPIO.input(IN_PIN)
                if reading != 0:
                    print()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
    # 	p1.stop()
        GPIO.cleanup()

    def choir(self):
        GPIO.setup(OUT_PIN, GPIO.OUT)
        GPIO.setup(IN_PIN, GPIO.OUT)

        p1 = GPIO.PWM(OUT_PIN, 280)
        p2 = GPIO.PWM(IN_PIN, 380)

        p1.start(5)
        p2.start(2.5)
        time.sleep(1)
        p1.stop()

        time.sleep(1.2)

        p1.ChangeFrequency(490)
        p1.start(5)
        time.sleep(1)

        p1.stop()
        p2.stop()

    # 	p1.start(0.2)
    # 	time.sleep(0.5)
    # 	p1.stop()

    # 	time.sleep(0.5)

    # 	p1.ChangeFrequency(255)
    # 	p1.start(1)
    # 	time.sleep(1)
    # 	p1.stop()

        #p1 = GPIO.PWM(OUT_PIN, 250)

        GPIO.cleanup()

    def notes(self):
        p = GPIO.PWM(OUT_PIN, 250)  # channel=12 frequency=50Hz
        p.start(0)
        try:
            while 1:
                for dc in numpy.arange(0.0, 11.0, 0.1):
                    p.ChangeDutyCycle(dc)
                    time.sleep(0.1)
                for dc in numpy.arange(10.0, -1.0, -1.0):
                    p.ChangeDutyCycle(dc)
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        p.stop()
        GPIO.cleanup()



@flaskapp.route("/home", methods=['GET'])
def dashboard():
    return render_template("home.html")

@flaskapp.route("/current_values", methods=['GET'])
def current_values():
    resp = dict()

    if request.args.get('x_from') is not None:
        x_from = int(request.form['x_from'])
    else:
        x_from = 0

    # app.series
    data = app.meter.chart_data
    labels = app.meter.chart_labels

    resp['data'] = data[-100:] #list(map(lambda i: float(i), data))
    resp['labels'] = labels[-100:]
    resp['start_at'] = int(labels[0])

    return str(json.dumps(resp))

@flaskapp.route("/switch", methods=['GET'])
def switch():
    if request.args.get('param') is not None:
        param_name = request.form['param']
        if param_name == "emitter":
            app.enable_emitter = not app.enable_emitter

    flags = dict()

    flags["enable_emitter"] = app.enable_emitter
    flags["enable_meter"] = app.enable_meter

    return str(json.dumps(flags))

main_thread = threading.currentThread()
app = SonarApp(main_thread)
app.sonar()

flaskapp.run(port=16000, host='192.168.1.135', threaded=True)

# notes()
#choir()
# listen()
