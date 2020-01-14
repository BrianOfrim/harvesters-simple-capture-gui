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

import s3_util

flags.DEFINE_string(
    "gentl_producer_path",
    "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti",
    "Path to the GenTL producer .cti file to use.",
)

flags.DEFINE_integer(
    "display_width", 1080, "Target image width for the display window.",
)

flags.DEFINE_integer("frame_rate", 30, "Frame rate to acquire images at.")

flags.DEFINE_string("image_dir", "../images", "The directory to save images to.")

flags.DEFINE_string("image_file_type", "jpg", "File type to save images as.")


flags.DEFINE_string("s3_bucket_name", None, "S3 bucket to send images to.")

flags.DEFINE_string("s3_image_dir", "data/images", "Prefix of the s3 image objects.")

WINDOW_NAME = "Acquire and Display"
exit_event = threading.Event()


class AcquiredImage:
    BORDER_COLOR = (3, 252, 53)
    BORDER_WIDTH = 10

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
        return cv2.copyMakeBorder(
            (
                self.get_resized_image(target_width)
                if target_width is not None
                else self.get_data()
            )[
                self.BORDER_WIDTH : -self.BORDER_WIDTH,
                self.BORDER_WIDTH : -self.BORDER_WIDTH,
            ],
            top=self.BORDER_WIDTH,
            bottom=self.BORDER_WIDTH,
            left=self.BORDER_WIDTH,
            right=self.BORDER_WIDTH,
            borderType=cv2.BORDER_ISOLATED,
            value=self.BORDER_COLOR,
        )

    def save(self, file_path: str) -> bool:
        try:
            cv2.imwrite(file_path, self.get_data())
        except:
            return False
        return True


def create_output_dir(dir_name) -> bool:
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        print("Creating output directory: %s" % dir_name)
        try:
            os.makedirs(dir_name)
        except OSError:
            print("Creation of the directory %s failed" % dir_name)
            return False
        else:
            print("Successfully created the directory %s " % dir_name)
            return True
    else:
        print("Output directory exists.")
        return True


def acquire_images(cam, image_queue: queue.Queue) -> None:
    cam.start_image_acquisition()
    print("Acquisition started.")
    while not exit_event.is_set():
        with cam.fetch_buffer() as buffer:
            component = buffer.payload.components[0]
            width = component.width
            height = component.height
            data_format = component.data_format
            image_data = component.data.copy()
            # clear stale image
            try:
                image_queue.get_nowait()
            except queue.Empty:
                pass
            # queue newest image
            image_queue.put(AcquiredImage(width, height, data_format, image_data))
    cam.stop_image_acquisition()
    print("Acquisition Ended.")


def save_images(save_queue: queue.Queue, use_s3: bool) -> None:
    try:
        while True:
            image = save_queue.get(block=True)
            if image is None:
                break
            file_path = os.path.join(
                flags.FLAGS.image_dir,
                "%i.%s" % (time.time(), flags.FLAGS.image_file_type),
            )
            save_successfull = image.save(file_path)
            print("Image saved at: %s" % file_path)

            if use_s3 and save_successfull:
                s3_util.upload_files(
                    flags.FLAGS.s3_bucket_name, [file_path], flags.FLAGS.s3_image_dir,
                )
    finally:
        print("Saving complete.")


def display_images(acquisition_queue: queue.Queue, save_queue: queue.Queue) -> None:

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    try:
        while True:

            retrieved_image = acquisition_queue.get(block=True)
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
                save_queue.put(retrieved_image)
                cv2.waitKey(500)
    finally:
        cv2.destroyAllWindows()
        exit_event.set()
        save_queue.put(None)


def apply_camera_settings(cam) -> None:
    cam.remote_device.node_map.AcquisitionFrameRateEnable.value = True
    cam.remote_device.node_map.AcquisitionFrameRate.value = min(
        flags.FLAGS.frame_rate, cam.remote_device.node_map.AcquisitionFrameRate.max
    )
    print(
        "Acquisition frame rate set to: %3.1f"
        % cam.remote_device.node_map.AcquisitionFrameRate.value
    )


def main(unused_argv):

    if not create_output_dir(flags.FLAGS.image_dir):
        print("Cannot create output annotations directory.")
        return

    use_s3 = True if flags.FLAGS.s3_bucket_name is not None else False

    if use_s3:
        if not s3_util.s3_bucket_exists(flags.FLAGS.s3_bucket_name):
            use_s3 = False
            print(
                "Bucket: %s either does not exist or you do not have access to it"
                % flags.FLAGS.s3_bucket_name
            )
        else:
            print(
                "Bucket: %s exists and you have access to it"
                % flags.FLAGS.s3_bucket_name
            )

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

    # Newest only single image queue
    acquisition_queue = queue.Queue(1)
    save_queue = queue.Queue()

    acquire_thread = threading.Thread(
        target=acquire_images, args=(cam, acquisition_queue,)
    )
    process_thread = threading.Thread(
        target=display_images, args=(acquisition_queue, save_queue,)
    )
    save_thread = threading.Thread(target=save_images, args=(save_queue, use_s3,))

    save_thread.start()
    process_thread.start()
    acquire_thread.start()

    process_thread.join()
    acquire_thread.join()
    save_thread.join()

    # clean up
    cam.destroy()
    h.reset()

    print("Exiting.")


if __name__ == "__main__":
    app.run(main)
