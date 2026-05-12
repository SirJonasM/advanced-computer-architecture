#set text(size: 11pt, font: "New Computer Modern")

#let course = "Advanced Computer Architecture"
#let group = "Übung 1 Group 05"
#let date = datetime.today()
#let authors = (
  "Josef Aktan",
  "Jonas Moewes",
  "Markus Steinle",
)

#set page(numbering: "(i)", header: [
  #set text(8pt)
  #grid(
    columns: 2,
    column-gutter: 1fr,
    row-gutter: 0.5em,
    align: (left, right),
    course, [University Heidelberg],
    group, date.display("[day].[month].[year]"),
  )
])

#align(right, text(10pt)[
  #authors.join(linebreak())
])

#align(center, title("Homework 1: Multithreaded AlphaBot2" ))

= Introduction
This experiment details the implementation of a multithreaded control system for the AlphaBot2 robot.
To achieve autonomous navigation and environment interaction, the robot executes three concurrent tasks:
+ Continuous Line Following: Real-time navigation along a black track.
+ Infrared-Based Obstacle Detection:
  Proximity sensing coupled with a buzzer signal that follows an incremental 1–2–3 pattern.
+ Camera-Based Object Recognition:
  Image classification using a neural network to update an RGB LED based on detected objects (Shoe $->$ Red, Bottle $->$ Yellow, Mug $->$ Green).
The performance of this multithreaded Python 3 implementation is evaluated against a hypothetical sequential model, specifically focusing on round-trip timing and system responsiveness.

= Optimization Goal
The primary limitation of a sequential architecture is the high computational latency of image processing.
In a single-threaded loop, the robot is unable to update its steering motors while the CPU is occupied with neural network inference, leading to frequent departures from the track.

Multithreading addresses this by decoupling high-latency tasks (object recognition) from time-critical tasks (motor control).
This approach leverages the Python thread scheduler to manage context switching automatically, ensuring the PID controller for line following remains active even during intensive vision calculations.

= Design Implementation
== Sequential Architecture
The sequential model utilizes a single execution loop that calls tasks in a fixed order:
+ Line Following:
  Invokes the follow_line method, which utilizes a PID controller to adjust motor speeds based on sensor input.
+ Obstacle Detection:
  Polls sensors and manages a state machine to detect "rising edges," ensuring the buzzer triggers only upon the initial detection of a new object.
+ Object Recognition:
  Captures a frame, classifies it, and updates the LED state. The LED remains active until a different target object is identified.

*Critical Flaw:* The "blocking" nature of the recognition step creates a "stop-and-go" motion, as the robot cannot adjust its heading until the image classification is complete.

== Multithreading Architecture
To facilitate concurrency, the AlphaBot2Multithreaded class was developed, inheriting from the base `AlphaBot2` class.
This architecture distributes the workload across three dedicated threads:
- Line-Following Thread (`line_following_thread`):
  Executes the PID control loop at a high frequency to maintain smooth tracking.
- Obstacle-Detection Thread (`obstacle_buzzer_thread`):
  Asynchronously polls IR sensors and manages the 1-2-3 buzzer logic independently of movement.
- Object-Recognition Thread (`recognition_thread`):
  Periodically captures camera frames and processes them using a quantized MobileNetV2 model.
  By running this in a separate thread, the heavy inference overhead does not "starve" the motor controllers.

Note: Each thread incorporates a brief sleep interval (yield) to prevent CPU saturation.

= Improvements
The performance gap between the two architectures is most apparent during multi-tasking.
In the sequential approach, the robot follows the line effectively only when that task is isolated.
Once obstacle detection and object recognition are added, the "blocking" nature of image processing creates significant latency.
During these processing intervals, the robot fails to update its steering, causing it to veer off the track.

In contrast, the multithreaded approach maintains path stability by decoupling motor control from vision processing.
Because the PID controller operates in its own thread, the robot continues to track the line accurately even while the neural network performs inference in the background.
This concurrency ensures that high-latency tasks no longer compromise the real-time responsiveness required for navigation, allowing the AlphaBot2 to execute all three tasks without departing from the track
This in the multithreaded approach it the can hold the line and does not veer off as easily with object detection and object recognition enabled.

= Conclusion
The experiment demonstrates that for real-time robotic applications, a multithreaded architecture is superior to a sequential one.
While the sequential model is simpler to implement, it creates a computational bottleneck where high-latency tasks specifically the MobileNetV2 image inference block time-critical navigation logic.
This results in intermittent steering failures and a loss of track autonomy.

By leveraging Python’s threading capabilities, the AlphaBot2 successfully decouples its motor control from its heavy processing tasks.
The thread scheduler ensures the PID controller receives the necessary CPU cycles to maintain path stability, even while concurrent IR polling and object recognition occur in the background.
Ultimately, multithreading transforms the robot from a "stop-and-go" machine into a fluid, responsive system capable of handling complex, simultaneous workloads without compromising its primary navigation goals.
