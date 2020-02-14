"""
Component that will perform object detection via tensorflow-lite-rest-server
"""
import datetime
import io
import json
import logging
import os
from datetime import timedelta
from typing import List, Tuple

import requests
from PIL import Image, ImageDraw
from homeassistant.util.pil import draw_box

import deepstack.core as ds
import homeassistant.helpers.config_validation as cv
import homeassistant.util.dt as dt_util
import voluptuous as vol
from homeassistant.components.image_processing import (
    ATTR_CONFIDENCE,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    DOMAIN,
    PLATFORM_SCHEMA,
    ImageProcessingEntity,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_NAME,
    CONF_IP_ADDRESS,
    CONF_PORT,
    HTTP_BAD_REQUEST,
    HTTP_OK,
    HTTP_UNAUTHORIZED,
)
from homeassistant.core import split_entity_id

_LOGGER = logging.getLogger(__name__)

CONF_SAVE_FILE_FOLDER = "save_file_folder"
CONF_TARGET = "target"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_PORT = 5000
DEFAULT_TARGET = "person"
RED = (255, 0, 0)
SCAN_INTERVAL = timedelta(days=365)  # NEVER SCAN.


PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_IP_ADDRESS): cv.string,
        vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
        vol.Optional(CONF_TARGET, default=DEFAULT_TARGET): cv.string,
        vol.Optional(CONF_SAVE_FILE_FOLDER): cv.isdir,
    }
)


def get_target(predictions: List, target: str):
    """
    Return only the info for the targets.
    """
    targets = []
    for result in predictions:
        if result["name"] == target:
            targets.append(result)
    return targets


def setup_platform(hass, config, add_devices, discovery_info=None):
    """Set up the classifier."""
    save_file_folder = config.get(CONF_SAVE_FILE_FOLDER)
    if save_file_folder:
        save_file_folder = os.path.join(save_file_folder, "")  # If no trailing / add it

    entities = []
    for camera in config[CONF_SOURCE]:
        object_entity = ObjectDetectEntity(
            config.get(CONF_IP_ADDRESS),
            config.get(CONF_PORT),
            config.get(CONF_TARGET),
            config.get(ATTR_CONFIDENCE),
            save_file_folder,
            camera.get(CONF_ENTITY_ID),
            camera.get(CONF_NAME),
        )
        entities.append(object_entity)
    add_devices(entities)


class ObjectDetectEntity(ImageProcessingEntity):
    """Perform a face classification."""

    def __init__(
        self,
        ip_address,
        port,
        target,
        confidence,
        save_file_folder,
        camera_entity,
        name=None,
    ):
        """Init with the API key and model id."""
        super().__init__()
        self._object_detection_url = f"http://{ip_address}:{port}/v1/object/detection"
        self._target = target
        self._confidence = confidence
        self._camera = camera_entity
        if name:
            self._name = name
        else:
            camera_name = split_entity_id(camera_entity)[1]
            self._name = "tflite_{}".format(camera_name)
        self._state = None
        self._targets = []
        self._last_detection = None

        if save_file_folder:
            self._save_file_folder = save_file_folder

    def process_image(self, image):
        """Process an image."""
        self._image_width, self._image_height = Image.open(
            io.BytesIO(bytearray(image))
        ).size
        self._state = None
        self._targets = []

        payload = {"image": image}
        response = requests.post(self._object_detection_url, files=payload)
        if not response.status_code == HTTP_OK:
            return

        predictions = response.json()
        self._targets = get_target(predictions["objects"], self._target)
        self._state = len(self._targets)
        if hasattr(self, "_save_file_folder") and self._state > 0:
            self.save_image(image, self._targets, self._target, self._save_file_folder)

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera

    @property
    def state(self):
        """Return the state of the entity."""
        return self._state

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def unit_of_measurement(self):
        """Return the unit of measurement."""
        target = self._target
        if self._state != None and self._state > 1:
            target += "s"
        return target

    @property
    def device_state_attributes(self):
        """Return device specific state attributes."""
        attr = {}
        if self._targets:
            attr["targets"] = [result["score"] for result in self._targets]
        if self._last_detection:
            attr["last_{}_detection".format(self._target)] = self._last_detection
        return attr

    def save_image(self, image, predictions, target, directory):
        """Save a timestamped image with bounding boxes around targets."""
        img = Image.open(io.BytesIO(bytearray(image))).convert("RGB")
        draw = ImageDraw.Draw(img)

        for prediction in predictions:
            prediction_confidence = ds.format_confidence(prediction["score"])
            if (
                prediction["name"] in target
                and prediction_confidence >= self._confidence
            ):
                draw_box(
                    draw,
                    prediction['box'],
                    self._image_width,
                    self._image_height,
                    text=str(prediction_confidence),
                    color=RED,
                )

        latest_save_path = directory + "{}_latest_{}.jpg".format(self._name, target[0])
        img.save(latest_save_path)
