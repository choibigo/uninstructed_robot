import sys
sys.path.append(r'D:\workspace\Difficult\git\OmniGibson')

def get():
    root_path = r"D:\workspace\Difficult\git\OmniGibson\omnigibson\data\og_dataset\objects"
    result_dict = {}

    for category in os.listdir(root_path):
        result_dict[category] = []

        for model in os.listdir(os.path.join(root_path, category)):
            result_dict[category].append(model)


    result_dict.pop('banana')
    return result_dict


def model_category(category_models):

    model_category_dict = {}
    
    for category, models in category_models:
        for model in models:
            model_category_dict[model] = category

    return model_category_dict

if __name__ =="__main__":

    print(len(model_category(get())))

