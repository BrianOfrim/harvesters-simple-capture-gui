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

flags.DEFINE_integer(
    "display_width", 640, "Target image width for the display window.",
)

flags.DEFINE_integer("frame_rate", 30, "Frame rate to acquire images at.")

WINDOW_NAME = "Acquire and Display"
BOARDER_COLOR = (3, 252, 53)

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

    def process_image(self) -> None:
        if self.processed:
            return

        if self.data_format in mono_location_formats:
            self.image_data = self.image_data.reshape(self.height, self.width)
            self.processed = True
        elif self.data_format == "BayerRG8":
            self.image_data = cv2.cvtColor(
                self.image_data.reshape(self.height, self.width), cv2.COLOR_BayerRG2RGB
            )
            self.data_format == "RGB8"
            self.processed = True
        elif self.data_format in rgb_formats or self.data_format in bgr_formats:

            self.image_data = self.image_data.reshape(self.height, self.width, 3)

            if self.data_format in bgr_formats:
                # Swap every R and B:
                content = content[:, :, ::-1]
            self.processed = True
        else:
            print("Unsupported pixel format: %s" % self.data_format)

    def get_resized_image(self, target_width: int) -> np.ndarray:
        resize_ratio = float(target_width / self.width)
        return cv2.resize(self.image_data, (0, 0), fx=resize_ratio, fy=resize_ratio)

    def get_highlighted_image(self, target_width: int = None) -> np.ndarray:
        bordersize = 10
        border = cv2.copyMakeBorder(
            (
                self.get_resized_image(target_width)
                if target_width is not None
                else self.get_data()
            )[bordersize:-bordersize, bordersize:-bordersize],
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_ISOLATED,
            value=BOARDER_COLOR,
        )
        return border


def acquire_images(cam, image_queue) -> None:
    cam.start_image_acquisition()
    while not exit_event.is_set():
        with cam.fetch_buffer() as buffer:
            # only queue a new image when display thread asks for one by blocking on get
            if image_queue.empty():
                component = buffer.payload.components[0]
                width = component.width
                height = component.height
                data_format = component.data_format
                image_data = component.data.copy()
                image_queue.put(RetrievedImage(width, height, data_format, image_data))
    cam.stop_image_acquisition()


def display_images(image_queue) -> None:

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    try:
        while True:

            retrieved_image = image_queue.get(block=True)
            if retrieved_image is None:
                break

            retrieved_image.process_image()

            # copy the image for modification and display
            display_image_data = retrieved_image.get_resized_image(
                flags.FLAGS.display_width
            )

            cv2.imshow(WINDOW_NAME, display_image_data)
            keypress = cv2.waitKey(1)

            if keypress == 27:
                # escape key pressed
                break
            elif cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                # x button clicked
                break
            elif keypress == 13:
                # Enter key pressed
                cv2.imshow(
                    WINDOW_NAME,
                    retrieved_image.get_highlighted_image(flags.FLAGS.display_width),
                )
                cv2.waitKey(500)
    finally:
        cv2.destroyAllWindows()
        exit_event.set()


def apply_camera_settings(cam) -> None:
    cam.remote_device.node_map.AcquisitionFrameRateEnable = True
    cam.remote_device.node_map.AcquisitionFrameRate = flags.FLAGS.frame_rate


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

    apply_camera_settings(cam)

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
