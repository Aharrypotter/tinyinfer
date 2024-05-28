import os
from tempfile import TemporaryDirectory

from .CodeGenerator import CodeGenerator
from .GeneralMemoryScheduler import GeneralMemoryScheduler
from .TfliteConvertor import TfliteConvertor


def GenerateSourceFilesFromTFlite(
    tflite_path,
    life_cycle_path=None,
):
    use_inplace = True

    with TemporaryDirectory() as WORKING_DIR:
        if life_cycle_path is None:
            schedule_image_path = os.path.join(WORKING_DIR, "schedule.png")
        else:
            schedule_image_path = life_cycle_path

        tf_convertor = TfliteConvertor(tflite_path)
        tf_convertor.parseOperatorInfo()
        layer = tf_convertor.layer
        outTable = []
        VisaulizeTrainable = False  # disable for code gen
        memory_scheduler = GeneralMemoryScheduler(
            layer,
            False,
            False,
            outputTables=outTable,
            inplace=use_inplace,
            mem_visual_path=schedule_image_path,
            VisaulizeTrainable=VisaulizeTrainable,
        )
        memory_scheduler.USE_INPLACE = use_inplace
        memory_scheduler.allocateMemory()

        outTable = tf_convertor.outputTables if hasattr(
            tf_convertor, "outputTables") else []
        code_generator = CodeGenerator(
            memsche=memory_scheduler,
            inplace=memory_scheduler.USE_INPLACE,
            unsigned_input=False,
            patch_params=None,
            FP_output=False,
            profile_mode=False,
            fp_requantize=True,
            tflite_op=False,
            dummy_address=False,
            outputTables=outTable,
        )
        # set detection outputs before codegen if any
        code_generator.codeGeneration()

        return memory_scheduler.buffers["input_output"]
