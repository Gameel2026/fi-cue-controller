#!/bin/sh
# Rebuild for Cortex-M4F with LLVM (clang 18, ld.lld). Output: fi_mcu.elf, flash.bin, symbols.txt, size.txt
CF="--target=thumbv7em-none-eabihf -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -ffreestanding -fno-builtin -fno-math-errno -nostdlib -Wall"
clang $CF -fstack-usage -c fi_mcu.c -o fi_mcu.o
ld.lld -T link.ld --entry=run -o fi_mcu.elf fi_mcu.o
llvm-objcopy -O binary --only-section=.text --only-section=.rodata fi_mcu.elf flash.bin
llvm-nm fi_mcu.elf > symbols.txt
llvm-size -A fi_mcu.elf > size.txt
