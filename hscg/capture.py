import copy
import os
import time
import threading
import typing
import queue

from absl import app, flags
from cv2 import cv2
from harvesters.core import Harvester
from harvesters.util.pfnc import (
    mono_location_formats,
    bayer_location_formats,
    rgb_formats,
    bgr_formats,
)
import numpy as np

flags.DEFINE_string(
    "gentl_producer_path",
    "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti",
    "Path to the GenTL producer .cti file to use.",
)

flags.DEFINE_float(
    "display_scale_factor", 0.5, "Scale factor to apply to displayed images",
)

WINDOW_NAME = "Acquire and Display"

exit_event = threading.Event()


class RetrievedImage:
    def __init__(
        self, width: int, height: int, data_format: str, image_data: np.ndarray
    ):
        self.width: int = width
        self.height: int = height
        self.data_format: str = data_format
        self.image_data: np.ndarray = image_data
        self.processed = False

    def get_data(self, process: bool = False) -> np.ndarray:
        if process:
            self.process_image()
        return self.image_data

    def process_image(self):
        if self.processed:
            return
        if self.data_format in mono_location_formats:
            self.image_data = self.image_data.reshape(self.height, self.width)
            self.processed = True
        if self.data_format == "BayerRG8":
            self.image_data = cv2.cvtColor(
                self.image_data.reshape(self.height, self.width), cv2.COLOR_BayerRG2RGB
            )
            self.data_format == "RGB8"
            self.processed = True
        else:
            if self.data_format in rgb_formats or self.data_format in bgr_formats:

                self.image_data = self.image_data.reshape(self.height, self.width, 3)

                if self.data_format in bgr_formats:
                    # Swap every R and B:
                    content = content[:, :, ::-1]
                self.processed = True
            else:
                print("Unsupported pixel format: %s" % self.data_format)


def acquire_images(cam, image_queue) -> None:
    cam.start_image_acquisition()
    while not exit_event.is_set():
        with cam.fetch_buffer() as buffer:
            component = buffer.payload.components[0]
            width = component.width
            height = component.height
            data_format = component.data_format
            image_data = component.data.copy()
            # only queue a new image when display thread asks for one by blocking on get
            if image_queue.empty():
                image_queue.put(RetrievedImage(width, height, data_format, image_data))
    cam.stop_image_acquisition()


def display_images(image_queue) -> None:

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    while 1:
        retrieved_image = image_queue.get(block=True)
        if retrieved_image is None:
            break

        cv2.imshow(WINDOW_NAME, retrieved_image.get_data(process=True))
        keypress = cv2.waitKey(1)

        if keypress == 27:
            # escape key pressed
            break

    cv2.destroyAllWindows()
    exit_event.set()


def main(unused_argv):
    h = Harvester()
    h.add_cti_file(flags.FLAGS.gentl_producer_path)
    if len(h.cti_files) == 0:
        print("No valid cti file found at %s" % flags.FLAGS.gentl_producer_path)
        h.reset()
        return
    print("Currently available genTL Producer CTI files: ", h.cti_files)

    h.update_device_info_list()
    if len(h.device_info_list) == 0:
        print("No compatible devices detected.")
        h.reset()
        return

    print("Available devices List: ", h.device_info_list)
    print("Using device: ", h.device_info_list[0])

    cam = h.create_image_acquirer(list_index=0)

    image_queue = queue.Queue()

    acquire_thread = threading.Thread(target=acquire_images, args=(cam, image_queue,))
    process_thread = threading.Thread(target=display_images, args=(image_queue,))

    process_thread.start()
    acquire_thread.start()

    process_thread.join()
    acquire_thread.join()

    # clean up
    cam.destroy()
    h.reset()


if __name__ == "__main__":
    app.run(main)
