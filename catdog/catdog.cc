#include <stdio.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>
#include <random>
#include <chrono>

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

int main()
{
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("model/catdog.tflite");

    if (!model) {
        printf("Failed to mmap model\n");
        return -1;
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

    printf("invoke time: %lld ms\n", elapsed.count());

    const std::vector<int> &t_outputs = interpreter->outputs();
    TfLiteTensor *tensor_output = interpreter->tensor(t_outputs[0]);

    auto cat_score = tensor_output->data.f[0];
    auto dog_score = tensor_output->data.f[1];

    printf("cat_score: %f, dog_score=%f, result=%s\n", cat_score, dog_score, ((cat_score > dog_score) ? "Cat" : "Dog"));

    return 0;
}
