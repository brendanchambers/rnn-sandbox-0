__author__ = 'Brendan'

import mido
import rtmidi_python as rtmidi

midi_out = rtmidi.MidiOut()
for port_name in midi_out.ports:
    print port_name


midi_in = rtmidi.MidiIn()
midi_in.open_port(0)

while True:
    message, delta_time = midi_in.get_message()
    if message:
        print message, delta_time


with mido.open_input('nanoKontrol2') as inport:
    for msg in inport:
        print(msg)