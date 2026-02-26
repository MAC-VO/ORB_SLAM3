#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FORCE_REBUILD=false
if [[ "${1:-}" == "force" ]]; then
    FORCE_REBUILD=true
    shift
fi

if command -v nproc >/dev/null 2>&1; then
    BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
else
    BUILD_JOBS="${BUILD_JOBS:-1}"
fi

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required command '$command_name' is not available in PATH." >&2
        exit 1
    fi
}

require_directory() {
    local directory_path="$1"
    if [[ ! -d "$directory_path" ]]; then
        echo "Error: required directory does not exist: $directory_path" >&2
        exit 1
    fi
}

require_file() {
    local file_path="$1"
    if [[ ! -f "$file_path" ]]; then
        echo "Error: required file does not exist: $file_path" >&2
        exit 1
    fi
}

ensure_orb_vocabulary() {
    local vocabulary_dir="$ROOT_DIR/Vocabulary"
    local vocabulary_text="$vocabulary_dir/ORBvoc.txt"
    local vocabulary_archive="$vocabulary_dir/ORBvoc.txt.tar.gz"

    if [[ -f "$vocabulary_text" ]]; then
        echo "[ORB_SLAM3] Vocabulary already extracted, skipping."
        return
    fi

    require_command tar
    require_file "$vocabulary_archive"

    echo "[ORB_SLAM3] Extracting ORBvoc.txt from archive..."
    tar -xzf "$vocabulary_archive" -C "$vocabulary_dir"

    if [[ ! -f "$vocabulary_text" ]]; then
        echo "Error: extraction did not produce expected file: $vocabulary_text" >&2
        exit 1
    fi
}

clean_build_caches() {
    echo "[ORB_SLAM3] Force rebuild: removing all build caches..."
    rm -rf "$ROOT_DIR/Thirdparty/DBoW2/build"
    rm -rf "$ROOT_DIR/Thirdparty/g2o/build"
    rm -rf "$ROOT_DIR/Thirdparty/Sophus/build"
    rm -rf "$ROOT_DIR/build"
}

build_with_cmake() {
    local source_dir="$1"
    local build_dir="$2"
    local target_name="$3"
    shift 3

    echo "[ORB_SLAM3] Configuring ${target_name}..."
    cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release "$@"

    echo "[ORB_SLAM3] Building ${target_name}..."
    cmake --build "$build_dir" -- -j"${BUILD_JOBS}"
}

main() {
    if [[ "$FORCE_REBUILD" == true ]]; then
        clean_build_caches
    fi

    require_command cmake

    require_directory "$ROOT_DIR/Thirdparty/DBoW2"
    require_directory "$ROOT_DIR/Thirdparty/g2o"
    require_directory "$ROOT_DIR/Thirdparty/Sophus"
    require_directory "$ROOT_DIR/Vocabulary"
    ensure_orb_vocabulary

    build_with_cmake "$ROOT_DIR/Thirdparty/DBoW2" "$ROOT_DIR/Thirdparty/DBoW2/build" "Thirdparty/DBoW2"
    build_with_cmake "$ROOT_DIR/Thirdparty/g2o" "$ROOT_DIR/Thirdparty/g2o/build" "Thirdparty/g2o"
    build_with_cmake \
        "$ROOT_DIR/Thirdparty/Sophus" \
        "$ROOT_DIR/Thirdparty/Sophus/build" \
        "Thirdparty/Sophus" \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF


    build_with_cmake "$ROOT_DIR" "$ROOT_DIR/build" "ORB_SLAM3"

    if [[ ! -f "$ROOT_DIR/lib/libORB_SLAM3.so" ]]; then
        echo "Error: expected output library not found: $ROOT_DIR/lib/libORB_SLAM3.so" >&2
        exit 1
    fi

    if python3 -c "import pybind11" 2>/dev/null; then
        echo "[ORB_SLAM3] Building pybind11 Python bindings..."
        cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" \
            -DCMAKE_BUILD_TYPE=Release \
            -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
        cmake --build "$ROOT_DIR/build" --target orbslam3_python -- -j"${BUILD_JOBS}"
    else
        echo "[ORB_SLAM3] pybind11 not found, skipping Python bindings."
    fi

    echo "[ORB_SLAM3] Install build complete."
}

main "$@"
