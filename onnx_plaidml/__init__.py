# Copyright Vertex.AI.

__version_info__ = (0, 1, 0)
__version__ = '.'.join([str(v) for v in __version_info__])


class Error(Exception):
    pass


class DeviceNotFoundError(Error):

    def __init__(self, device_name, devices_available):
        self.device_name = device_name
        self.devices_available = devices_available
        if devices_available == []:
            devmsg = 'perhaps you need to run \'plaidml-setup\' to configure devices on this machine'
        elif len(devices_available) == 1:
            devmsg = 'the only available device is \'{}\''.format(devices_available[0])
        elif len(devices_available) == 2:
            devmsg = 'the available devices are \'{}\' and \'{}\''.format(
                devices_available[0], devices_available[1])
        else:
            devmsg = 'the available devices are {}, and \'{}\''.format(
                ', '.join(['\'{}\''.format(d)
                           for d in devices_available[:-1]]), devices_available[-1])
        super(DeviceNotFoundError, self).__init__('Unable to find device \'{}\'; {}'.format(
            device_name, devmsg))
