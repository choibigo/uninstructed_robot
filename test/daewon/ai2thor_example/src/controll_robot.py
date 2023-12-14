import os
import time
import ai2thor.controller
import ai2thor.video_controller
import prior
import cv2

def my_remote(direct):
    if direct == 'w':
      controller.step(dict(action='MoveAhead', moveMagnitude='0.5'))
    elif direct == 's':
      controller.step(dict(action='MoveBack', moveMagnitude='0.5'))
    elif direct == 'a':
      controller.step(dict(action='RotateLeft', degrees=10))
    elif direct == 'd':
      controller.step(dict(action='RotateRight', degrees=10))
    elif direct == 'u':
      controller.step(action="LookUp", degrees=10)
    elif direct == 'j':
      controller.step(action="LookDown", degrees=10)
    else:
        pass

dataset = prior.load_dataset("procthor-10k")


controller = ai2thor.controller.Controller()
# controller.reset(scene=dataset["train"][7],
controller.reset(scene=dataset["train"][8],
                width=1000,
                height=1000,)
controller.step(dict(action='Initialize', gridSize=0.25))


save_path = "./debug_image"


i=0
while True:
    print("Input: ")
    _in = input()
    my_remote(_in)
    controller.step(action="Done")

    image = controller.last_event.cv2img
    cv2.imwrite(os.path.join(save_path, f"{i}.png"), image)
    i+=1