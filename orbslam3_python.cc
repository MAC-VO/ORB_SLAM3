/**
 * Pybind11 bindings for ORB-SLAM3 System.
 *
 * Exposes a minimal Python API: construct, track_stereo, get_tracking_state,
 * shutdown.  Poses are returned as 7-element numpy arrays
 * [tx, ty, tz, qx, qy, qz, qw].
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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

    py::class_<ORB_SLAM3::System>(m, "System")
        .def(py::init([](const std::string &vocab_file,
                         const std::string &settings_file,
                         bool use_viewer) {
            return new ORB_SLAM3::System(vocab_file, settings_file,
                                         ORB_SLAM3::System::STEREO, use_viewer);
        }),
        py::arg("vocab_file"),
        py::arg("settings_file"),
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

        .def("shutdown",
             [](ORB_SLAM3::System &self) { self.Shutdown(); });
}
