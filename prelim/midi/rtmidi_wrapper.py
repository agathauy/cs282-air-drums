import time
import rtmidi

# midiout = rtmidi.MidiOut()
# available_ports = midiout.get_ports()

# # if available_ports:
# #     midiout.open_port(0)
# # else:
# midiout.open_virtual_port("My virtual output")


NOTE_ON = 0x90
NOTE_OFF = 0x80

class MidiOutWrapper:
    # https://github.com/SpotlightKid/python-rtmidi/issues/38
    # note = 36
    # channel = 10  # counting from one here
    # NOTE_ON = 0x90
    # status = NOTE_ON | (channel - 1)  # bit-wise OR of NOTE_ON and channel (zero-based)
    # midiout.send_message([status, note, 127)  # full velocity


    # midiout = MidiOuWrapper(midiout)  # passing in a rtmidi.MidiOut instance
    # midiout.channel_message(NOTE_ON, 36, 127, ch=10)
    # midiout.channel_message(NOTE_ON, 40, 127)  # sent on default channel 1

    # midiout.program_change(9)  # programs counted from zero too!
    # midiout.note_on(36, ch=10)  # default velocity 127
    # midiout.note_on(40, 64)  # default channel 1
    # midiout.note_on(60)  # default velocity and channel
    # midiout.note_off(36, ch=10)   # default release velocity 0
    # midiout.note_off(40, 64)  # default channel 1
    # midiout.note_off(60)  # default release velocity and channel

    def __init__(self, midi, ch=1):
        self.channel = ch
        self._midi = midi

    def channel_message(self, command, *data, ch=None):
        """Send a MIDI channel mode message."""
        command = (command & 0xf0) | ((ch if ch else self.channel) - 1 & 0xf)
        msg = [command] + [value & 0x7f for value in data]
        self._midi.send_message(msg)

    def note_off(self, note, velocity=0, ch=None):
        """Send a 'Note Off' message."""
        self.channel_message(NOTE_OFF, note, velocity, ch=ch)

    def note_on(self, note, velocity=127, ch=None):
        """Send a 'Note On' message."""
        self.channel_message(NOTE_ON, note, velocity, ch=ch)

    # def program_change(self, program, ch=None):
    #     """Send a 'Program Change' message."""
    #     self.channel_message(PROGRAM_CHANGE, program, ch=ch)



midiout = rtmidi.MidiOut()


midiout.open_virtual_port("AirDrums 2pts")


if __name__ == '__main__':

    # with midiout:
    print("In midiout")
    c_midiout = MidiOutWrapper(midiout)

    while True:

            # note_on = [0x90, 60, 112] # channel 1, middle C, velocity 112
            # note_off = [0x80, 60, 0]
            # midiout.send_message(note_on)
            # time.sleep(0.5)
            # midiout.send_message(note_off)
            # time.sleep(0.1)
            c_midiout.note_on(36, ch=10, velocity=120)
            time.sleep(0.01)
            c_midiout.note_off(36, ch=10)
        
del midiout
