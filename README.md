# HASS-tensorflow-lite-rest-server
Home Assistant integration for the [tensorflow-lite-rest-server](https://github.com/robmarkcole/tensorflow-lite-rest-server). Make sure the tflite server is running then configure this integration as below. This integration adds an `image_processing` entity whose state is the number of `target` objects identified, which defaults to `person` detection. This integration does not automatically scan an image, you must call the image_processing `scan` service to process an image 

## Home Assistant setup
Place the `custom_components` folder in your configuration directory (or add its contents to an existing `custom_components` folder). Then configure object detection. **Important:** It is necessary to configure only a single camera per `tflite_server` entity. If you want to process multiple cameras, you will therefore need multiple `tflite_server` `image_processing` entities.

Add to your Home-Assistant config:

```yaml
image_processing:
  - platform: tflite_server
    ip_address: localhost
    port: 5000
    # scan_interval: 30 # Optional, in seconds
    save_file_folder: /config/www/
    source:
      - entity_id: camera.local_file
```

Configuration variables:
- **ip_address**: the ip address of your tflite server.
- **port**: (Optional, default 5000 )the port of your tflite server.
- **save_file_folder**: (Optional) The folder to save processed images to. Note that folder path should be added to [whitelist_external_dirs](https://www.home-assistant.io/docs/configuration/basic/)
- **source**: Must be a camera.
- **target**: (Optional, default `person`) The target object class.
- **confidence**: (Optional) The confidence (in %) above which detected targets are counted in the sensor state. Default value: 80
- **name**: (Optional) A custom name for the the entity.