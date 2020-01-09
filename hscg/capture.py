import os
import time
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

flags.DEFINE_string(
    "gentl_producer_path",
    "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti",
    "Path to the GenTL producer .cti file to use.",
)

flags.DEFINE_float(
    "display_scale_factor", 0.5, "Scale factor to apply to displayed images",
)

WINDOW_NAME = "Acquire and Display"


def acquire_and_display_images(cam):
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

    cam.start_image_acquisition()

    image_count = 0

    while 1:
        with cam.fetch_buffer() as buffer:

            payload = buffer.payload
            component = payload.components[0]
            width = component.width
            height = component.height
            data_format = component.data_format
            if image_count % 100 == 0:
                print(
                    "Image %i width: %i height: %i format: %s"
                    % (image_count, width, height, data_format)
                )

            # Reshape the image so that it can be drawn on the VisPy canvas:
            if data_format in mono_location_formats:
                content = component.data.reshape(height, width)
            if data_format == "BayerRG8":
                content = cv2.cvtColor(
                    component.data.reshape(height, width), cv2.COLOR_BayerRG2RGB
                )
            else:
                # The image requires you to reshape it to draw it on the
                # canvas:
                if data_format in rgb_formats or data_format in bgr_formats:

                    content = component.data.reshape(
                        height,
                        width,
                        int(
                            component.num_components_per_pixel
                        ),  # Set of R, G, B, and Alpha
                    )

                    if data_format in bgr_formats:
                        # Swap every R and B:
                        content = content[:, :, ::-1]
                else:
                    print("Unsupported pixel format: %s" % data_format)
                    break

                if flags.FLAGS.display_scale_factor != 1:
                    content = cv2.resize(
                        content,
                        (0, 0),
                        fx=flags.FLAGS.display_scale_factor,
                        fy=flags.FLAGS.display_scale_factor,
                    )

            cv2.imshow(WINDOW_NAME, content)
            keypress = cv2.waitKey(1)
            image_count += 1

            if keypress == 27:
                # escape key pressed
                break

    print("Number of images grabbed: %i" % image_count)

    cam.stop_image_acquisition()
    cv2.destroyAllWindows()


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

    acquire_and_display_images(cam)

    # clean up
    cam.destroy()
    h.reset()


if __name__ == "__main__":
    app.run(main)
