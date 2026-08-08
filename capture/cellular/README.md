# capture/cellular

Passive LTE cell broadcast decode: Cell ID, PLMN/MCC-MNC, band, MIB, SIB1-3,
RSRP/RSRQ/SINR. Based on a fork of LTE-Cell-Scanner, ported onto libbladeRF2's native
AD9361 gain-control API for the bladeRF 2.0 micro xA9.

**Hard scope boundary**: broadcast-channel decode only (PSS/SSS/PBCH/PDSCH SIB). No
paging-channel decoding, no RRC connection setup, nothing that identifies or tracks
individual subscribers.

**Status**: not yet implemented. First milestone is a hardware-free DSP spike validating
the search/decode chain against recorded or synthetic LTE IQ files, followed by a
hardware spike confirming the gain-control port locks onto a real signal on physical
xA9 silicon.
