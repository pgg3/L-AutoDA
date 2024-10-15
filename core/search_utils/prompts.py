class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            "Given an image 'org_img', its adversarial image 'best_adv_img', "
            "and a random normal noise 'std_normal_noise', "
            "you need to design an algorithm to combine them to search for a new adversarial example 'x_new'. "
            "'hyperparams' ranges from 0.5 to 1.5.  It gets larger when "
            "this algorithm outputs more adversarial examples, and vice versa. "
            "It can be used to control the step size of the search."
            "Operations you may use include: adding, subtracting, multiplying, dividing, "
            "dot product, and l2 norm computation. Design an novel algorithm with various search techniques. Your code "
            "should be able to run without further assistance. "
        )
        self.prompt_func_name = "draw_proposals"
        self.prompt_func_inputs = ["org_img", "best_adv_img", "std_normal_noise", "hyperparams"]
        self.prompt_func_outputs = ["x_new"]
        self.prompt_inout_inf = (
            "'org_img', 'best_adv_img', 'x_new', and 'std_normal_noise' are shaped as (3, img_height, img_width). "
            "The bound of images are [0, 1]. "
            "'std_normal_noise' contains random normal noises. "
            "'hyperparams' is a numpy array with shape (1,). "
        )
        self.prompt_other_inf = ("All inouts are numpy arrays.")

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf
