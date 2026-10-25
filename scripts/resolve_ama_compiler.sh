#!/usr/bin/env bash
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Choose the C/C++ compiler the AMA Cryptography native build uses.
#
# AMA's CMake sets the floor: GCC >= 12 (INVARIANT-8). The image's default
# GCC/G++ is used when it meets that floor, then an explicit gcc-12/g++-12
# when the image ships one, and otherwise the apt install step provides
# gcc-12/g++-12 and the build calls them by absolute path.
#
# Prints key=value lines on stdout, in the form $GITHUB_OUTPUT expects:
#   cc, cxx, version, source (image-default | image-gcc-12 | apt-gcc-12),
#   need-apt (true when the install step must provide the compiler).
# A human-readable summary goes to stderr.
set -euo pipefail

floor="${AMA_GCC_FLOOR:-12}"

# Echo "<cc> <cxx> <version>" when both compilers exist and meet the floor.
pick() {
    local c="$1" x="$2" cmajor xmajor
    command -v "${c}" >/dev/null 2>&1 || return 1
    command -v "${x}" >/dev/null 2>&1 || return 1
    cmajor="$("${c}" -dumpfullversion | cut -d. -f1)"
    xmajor="$("${x}" -dumpfullversion | cut -d. -f1)"
    [ "${cmajor}" -ge "${floor}" ] && [ "${xmajor}" -ge "${floor}" ] || return 1
    echo "${c} ${x} $("${c}" -dumpfullversion)"
}

need_apt=false
if read -r cc cxx ver < <(pick gcc g++); then
    source=image-default
elif read -r cc cxx ver < <(pick gcc-12 g++-12); then
    source=image-gcc-12
else
    cc=/usr/bin/gcc-12
    cxx=/usr/bin/g++-12
    ver="${floor}"
    source=apt-gcc-12
    need_apt=true
fi

echo "cc=${cc}"
echo "cxx=${cxx}"
echo "version=${ver}"
echo "source=${source}"
echo "need-apt=${need_apt}"
echo "AMA C compiler: ${cc} (${ver}, from ${source})" >&2
