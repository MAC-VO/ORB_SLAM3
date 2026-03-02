/**
 * Pybind11 bindings for ORB-SLAM3 System.
 *
 * Exposes a minimal Python API: construct, track_stereo, get_tracking_state,
 * get_all_frame_poses, shutdown.  Poses are returned as 7-element numpy arrays
 * [tx, ty, tz, qx, qy, qz, qw].
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "System.h"

namespace py = pybind11;

namespace {

cv::Mat numpy_to_cvmat(py::array_t<uint8_t> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim == 3) {
        return cv::Mat(static_cast<int>(buf.shape[0]),
                       static_cast<int>(buf.shape[1]),
                       CV_8UC3,
                       buf.ptr).clone();
    } else if (buf.ndim == 2) {
        return cv::Mat(static_cast<int>(buf.shape[0]),
                       static_cast<int>(buf.shape[1]),
                       CV_8UC1,
                       buf.ptr).clone();
    }
    throw std::runtime_error("numpy_to_cvmat: expected 2-D (HxW) or 3-D (HxWx3) uint8 array");
}

}  // anonymous namespace

PYBIND11_MODULE(orbslam3_python, m) {
    m.doc() = "Minimal pybind11 bindings for ORB-SLAM3 System";

    py::class_<ORB_SLAM3::StereoDirectInitConfig>(m, "StereoInitConfig")
        .def(py::init<>())
        .def_readwrite("fx", &ORB_SLAM3::StereoDirectInitConfig::fx)
        .def_readwrite("fy", &ORB_SLAM3::StereoDirectInitConfig::fy)
        .def_readwrite("cx", &ORB_SLAM3::StereoDirectInitConfig::cx)
        .def_readwrite("cy", &ORB_SLAM3::StereoDirectInitConfig::cy)
        .def_readwrite("width", &ORB_SLAM3::StereoDirectInitConfig::width)
        .def_readwrite("height", &ORB_SLAM3::StereoDirectInitConfig::height)
        .def_readwrite("baseline", &ORB_SLAM3::StereoDirectInitConfig::baseline)
        .def_readwrite("fps", &ORB_SLAM3::StereoDirectInitConfig::fps)
        .def_readwrite("rgb", &ORB_SLAM3::StereoDirectInitConfig::rgb)
        .def_readwrite("stereo_th_depth", &ORB_SLAM3::StereoDirectInitConfig::stereoThDepth)
        .def_readwrite("orb_n_features", &ORB_SLAM3::StereoDirectInitConfig::orbNFeatures)
        .def_readwrite("orb_scale_factor", &ORB_SLAM3::StereoDirectInitConfig::orbScaleFactor)
        .def_readwrite("orb_n_levels", &ORB_SLAM3::StereoDirectInitConfig::orbNLevels)
        .def_readwrite("orb_ini_th_fast", &ORB_SLAM3::StereoDirectInitConfig::orbIniThFAST)
        .def_readwrite("orb_min_th_fast", &ORB_SLAM3::StereoDirectInitConfig::orbMinThFAST)
        .def_readwrite("th_far_points", &ORB_SLAM3::StereoDirectInitConfig::thFarPoints)
        .def_readwrite("atlas_load_file", &ORB_SLAM3::StereoDirectInitConfig::atlasLoadFile)
        .def_readwrite("atlas_save_file", &ORB_SLAM3::StereoDirectInitConfig::atlasSaveFile);

    py::class_<ORB_SLAM3::System>(m, "System")
        .def(py::init([](const std::string &vocab_file,
                         const ORB_SLAM3::StereoDirectInitConfig &stereo_init_config,
                         bool use_viewer) {
            return new ORB_SLAM3::System(vocab_file, stereo_init_config,
                                         ORB_SLAM3::System::STEREO, use_viewer);
        }),
        py::arg("vocab_file"),
        py::arg("stereo_init_config"),
        py::arg("use_viewer") = false)

        .def("track_stereo",
             [](ORB_SLAM3::System &self,
                py::array_t<uint8_t> imL,
                py::array_t<uint8_t> imR,
                double timestamp) -> py::object {
                 cv::Mat cvL = numpy_to_cvmat(imL);
                 cv::Mat cvR = numpy_to_cvmat(imR);

                 Sophus::SE3f Tcw = self.TrackStereo(cvL, cvR, timestamp);

                 // Empty rotation matrix signals tracking failure
                 if (Tcw.rotationMatrix().isZero(0)) {
                     return py::none();
                 }

                 Eigen::Vector3f    t = Tcw.translation();
                 Eigen::Quaternionf q = Tcw.unit_quaternion();

                 // [tx, ty, tz, qx, qy, qz, qw]
                 py::array_t<float> out(7);
                 auto ptr = out.mutable_unchecked<1>();
                 ptr(0) = t.x();  ptr(1) = t.y();  ptr(2) = t.z();
                 ptr(3) = q.x();  ptr(4) = q.y();  ptr(5) = q.z();  ptr(6) = q.w();

                 return out;
             },
             py::arg("imL"), py::arg("imR"), py::arg("timestamp"))

        .def("get_tracking_state",
             [](ORB_SLAM3::System &self) -> int {
                 return self.GetTrackingState();
             })

        // Returns Nx8 float64 array: [timestamp_ns, tx, ty, tz, qx, qy, qz, qw]
        // Pose convention: T_wc (world-from-camera) as written by SaveTrajectoryEuRoC.
        // Re-derives every frame's absolute pose from its reference keyframe,
        // capturing any loop-closure / BA corrections applied after tracking.
        // Lost frames are skipped (the returned N <= total frames tracked).
        .def("get_all_frame_poses",
             [](ORB_SLAM3::System &self) -> py::object {
                 // Write trajectory to a temp file, parse it back.
                 std::string tmpfile = std::tmpnam(nullptr);
                 self.SaveTrajectoryEuRoC(tmpfile);

                 std::ifstream ifs(tmpfile);
                 if (!ifs.is_open()) return py::none();

                 std::vector<std::array<double,8>> rows;
                 std::string line;
                 while (std::getline(ifs, line)) {
                     std::istringstream ss(line);
                     std::array<double,8> r;
                     if (ss >> r[0] >> r[1] >> r[2] >> r[3]
                            >> r[4] >> r[5] >> r[6] >> r[7]) {
                         rows.push_back(r);
                     }
                 }
                 ifs.close();
                 std::remove(tmpfile.c_str());

                 if (rows.empty()) return py::none();

                 py::array_t<double> out({static_cast<ssize_t>(rows.size()),
                                          static_cast<ssize_t>(8)});
                 auto buf = out.mutable_unchecked<2>();
                 for (size_t i = 0; i < rows.size(); ++i)
                     for (size_t j = 0; j < 8; ++j)
                         buf(i, j) = rows[i][j];

                 return out;
             })

        .def("shutdown",
             [](ORB_SLAM3::System &self) { self.Shutdown(); });
}
