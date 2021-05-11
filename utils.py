import base64


def decode_image_base64(data):
    """
    Returns decoded base64 image if successful, else returns None.

    :param data: base64 string
    """
    try:

        # Removing header that is encoded at the beginning of the data
        _, encoded = data.split(",", 1)

        img = base64.b64decode(encoded)

    # Incorrect encoding format received
    except base64.binascii.Error:
        pass
    # Exception thrown when spliting header fails
    except ValueError:
        pass
    else:
        return img
