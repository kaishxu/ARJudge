import os
import re
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default='')
    parser.add_argument("--swap", action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print(args.result_path)
    with open(args.result_path, "r") as infile:
        result_lst = json.load(infile)

    if args.swap:
        result_path = args.result_path.split(".json")[0] + "_swap.json"
        with open(result_path, "r") as infile:
            result_swap_lst = json.load(infile)

        correct = 0
        correct_swap = 0
        consistency = 0
        pattern = r"response (1|2) is better"
        for idx in range(len(result_lst)):
            is_correct = False
            winner = str(result_lst[idx]["winner"])
            tmp = re.findall(pattern, result_lst[idx]["model_output"].lower())

            if len(tmp) > 0:
                if tmp[-1] == winner:
                    correct += 1
                    is_correct = True
            result_lst[idx]["extracted_result"] = tmp
            result_lst[idx]["result"] = is_correct

            is_correct_swap = False
            winner = str(result_swap_lst[idx]["winner"])
            tmp = re.findall(pattern, result_swap_lst[idx]["model_output"].lower())

            if len(tmp) > 0:
                if tmp[-1] == winner:
                    correct_swap += 1
                    is_correct_swap = True
            result_swap_lst[idx]["extracted_result"] = tmp
            result_swap_lst[idx]["result"] = is_correct_swap

            if is_correct == is_correct_swap:
                consistency += 1

        # save results
        with open(args.result_path, "w") as outfile:
            json.dump(result_lst, outfile, indent=4)
        with open(result_path, "w") as outfile:
            json.dump(result_swap_lst, outfile, indent=4)

        print("Acc:", "{:.4f}".format(correct/len(result_lst)))
        print("Swap Acc:", "{:.4f}".format(correct_swap/len(result_lst)))
        print("Ave Acc:", "{:.4f}".format((correct+correct_swap)/2/len(result_lst)))
        print("Con:", "{:.4f}".format(consistency/len(result_lst)))

        # save metrics
        metric_path = "/".join(args.result_path.split("/")[:-1]) + "/all_metrics.json"
        if os.path.exists(metric_path):
            with open(metric_path, "r") as infile:
                metric_dict = json.load(infile)
            
            name = args.result_path.split(".json")[0].split("/")[-1]
            metric_dict[name] = {}
            metric_dict[name]["Acc"] = "{:.4f}".format(correct/len(result_lst))
            metric_dict[name]["Swap Acc"] = "{:.4f}".format(correct_swap/len(result_lst))
            metric_dict[name]["Ave Acc"] = "{:.4f}".format((correct+correct_swap)/2/len(result_lst))
            metric_dict[name]["Con"] = "{:.4f}".format(consistency/len(result_lst))
            
            with open(metric_path, "w") as outfile:
                json.dump(metric_dict, outfile, indent=4)
        else:
            metric_dict = {}
            name = args.result_path.split(".json")[0].split("/")[-1]
            metric_dict[name] = {}
            metric_dict[name]["Acc"] = "{:.4f}".format(correct/len(result_lst))
            metric_dict[name]["Swap Acc"] = "{:.4f}".format(correct_swap/len(result_lst))
            metric_dict[name]["Ave Acc"] = "{:.4f}".format((correct+correct_swap)/2/len(result_lst))
            metric_dict[name]["Con"] = "{:.4f}".format(consistency/len(result_lst))
            
            with open(metric_path, "w") as outfile:
                json.dump(metric_dict, outfile, indent=4)

    else:
        correct = 0
        pattern = r"response (1|2) is better"
        for idx in range(len(result_lst)):

            winner = str(result_lst[idx]["winner"])
            tmp = re.findall(pattern, result_lst[idx]["model_output"].lower())
            if len(tmp) > 0:
                if tmp[-1] == winner:
                    correct += 1

        print("Acc:", "{:.4f}".format(correct/len(result_lst)))