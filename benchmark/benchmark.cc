#include <stdio.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>
#include <random>
#include <chrono>
#include <vector>

using namespace tflite;

namespace {

static std::random_device rd;
static std::mt19937 s_gen(rd());

static bool fillRandomInputTensor(tflite::Interpreter& interpreter)
{
    tflite::Subgraph &subgraph = interpreter.primary_subgraph();
    auto plan = subgraph.execution_plan();
    auto nodes = subgraph.nodes_and_registration();
    auto plan0 = plan[0];
    auto node = nodes[plan0].first;
    auto inputs = node.inputs;
    const std::vector<int> &t_inputs = interpreter.inputs();
    TfLiteTensor *tensor_input = interpreter.tensor(t_inputs[0]);
    if (tensor_input == nullptr) {
        return false;
    }

    TfLiteType model_input_tensor_type = tensor_input->type;
    int input_dims = tensor_input->dims->size;
    int input_size = 1;
    for (int i = 0; i < input_dims; i++) {
        input_size *= tensor_input->dims->data[i];
    }
    for (int i = 0; i < input_size; i++) {
        std::uniform_int_distribution<> dist(0, 256);
        int rand_pixel = dist(s_gen);
        switch (model_input_tensor_type) {
        case kTfLiteFloat32:
            if (interpreter.typed_input_tensor<float>(0) != nullptr)
                interpreter.typed_input_tensor<float>(0)[i] = static_cast<float>(rand_pixel) / static_cast<float>(256);
            break;
        case kTfLiteInt32:
            if (interpreter.typed_input_tensor<int32_t>(0) != nullptr)
                interpreter.typed_input_tensor<int32_t>(0)[i] = rand_pixel;
            break;
        case kTfLiteUInt8:
            if (interpreter.typed_input_tensor<uint8_t>(0) != nullptr)
                interpreter.typed_input_tensor<uint8_t>(0)[i] = rand_pixel;
            break;
        case kTfLiteInt64:
            if (interpreter.typed_input_tensor<int64_t>(0) != nullptr)
                interpreter.typed_input_tensor<int64_t>(0)[i] = rand_pixel;
            break;
        case kTfLiteInt16:
            if (interpreter.typed_input_tensor<int16_t>(0) != nullptr)
                interpreter.typed_input_tensor<int16_t>(0)[i] = rand_pixel;
            break;
        case kTfLiteInt8:
            if (interpreter.typed_input_tensor<int8_t>(0) != nullptr)
                interpreter.typed_input_tensor<int8_t>(0)[i] = rand_pixel;
            break;
        case kTfLiteFloat64:
            if (interpreter.typed_input_tensor<double>(0) != nullptr)
                interpreter.typed_input_tensor<double>(0)[i] = static_cast<double>(rand_pixel) / static_cast<float>(256);
            break;
        case kTfLiteUInt64:
            if (interpreter.typed_input_tensor<uint64_t>(0) != nullptr)
                interpreter.typed_input_tensor<uint64_t>(0)[i] = rand_pixel;
            break;
        case kTfLiteUInt32:
            if (interpreter.typed_input_tensor<uint32_t>(0) != nullptr)
                interpreter.typed_input_tensor<uint32_t>(0)[i] = rand_pixel;
            break;
        default:
            break;
        }
    }

    return true;
}

} // end of anonymous namespace

uint64_t benchmark_model(const std::string& modelpath)
{
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelpath.c_str());

    if (!model) {
        printf("Failed to find model\n");
        return 0;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    // Resize input tensors, if desired.
    interpreter->AllocateTensors();

    fillRandomInputTensor(*interpreter.get());

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    interpreter->Invoke();
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //printf("%s: invoke time: %lld ms\n", modelpath.c_str(), elapsed.count());

    return elapsed.count();
}

int main() {

    std::vector<std::string> mediapipe_models = {
        "model/face_detection_short_range.tflite",
        "model/face_detection_full_range.tflite",
        "model/face_detection_full_range_sparse.tflite",
        "model/face_landmark.tflite",
        "model/face_landmark_with_attention.tflite",
        "model/pose_detection.tflite",
        "model/pose_landmark_lite.tflite",
        "model/pose_landmark_full.tflite",
        "model/pose_landmark_heavy.tflite",
        "model/palm_detection_lite.tflite",
        "model/palm_detection_full.tflite",
        "model/hand_landmark_lite.tflite",
        "model/hand_landmark_full.tflite",
        "model/hand_recrop.tflite",
        "model/iris_landmark.tflite",
        "model/object_detection_3d_cup.tflite",
        "model/object_detection_3d_sneakers_1stage.tflite",
        "model/object_detection_3d_sneakers.tflite",
        "model/selfie_segmentation.tflite",
        "model/selfie_segmentation_landscape.tflite",
    };

    for (const auto& model : mediapipe_models) {
        uint64_t sum = 0;
        int repeat = 10;

        for(int i=0; i < repeat; i++) {
            sum += benchmark_model(model);
        }

        double average = static_cast<double>(sum) / static_cast<double>(repeat);

        printf("%s: avg invoke time: %.2f ms\n", model.c_str(), average);
    }

    return 0;
}
