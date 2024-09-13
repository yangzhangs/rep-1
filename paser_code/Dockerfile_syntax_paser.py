import re

# 定义 Dockerfile 指令
import pymysql
from transformers import PreTrainedTokenizerFast

import threading

INSTRUCTIONS = [
    "FROM", "RUN", "CMD", "LABEL", "EXPOSE", "ENV", "ADD", "COPY",
    "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG", "ONBUILD",
    "STOPSIGNAL", "HEALTHCHECK", "SHELL", "MAINTAINER"
]

# 构建指令正则表达式模式
INSTRUCTION_PATTERN = re.compile(r"^\s*(?P<instruction>" + "|".join(INSTRUCTIONS) + r")\s*(?P<arguments>.*)", re.IGNORECASE)

tokenizer = PreTrainedTokenizerFast.from_pretrained("D:/research-related/niuniu work/dockerfill/tokenizer/dcc-tokenizer.json")
tokenizer.add_special_tokens({'bos_token': '<s>'})
tokenizer.add_special_tokens({'eos_token': '</s>'})
tokenizer.add_special_tokens({'unk_token': '<unk>'})
tokenizer.add_special_tokens({'mask_token': '<mask>'})
tokenizer.add_special_tokens({'pad_token': '<pad>'})


var = r"[a-zA-Z_][a-zA-Z0-9_]*"
str = r"([^$\n\s]|(\$("+var+r"\S+)|\${("+var+r"?P<var>\S+)}))*"
command = r"("+str+r"\S+)"
json_commands = r"\[("+command+r".+(,\s+("+command+r".+))*)\]"
key= r"("+str+r"\S+)"
value= r"("+str+r"\S+)"
#key_value = r"("+key+r"\S+)\s+("+value+r"\S+)|(("+key+r"\S+)=("+value+r"\S+))+"
port = r"(\d+(\/[a-zA-Z])?)+"
src = r"("+str+r"\S+)"
dest = r"("+str+r"\S+)"
user = r"("+str+r"\S+)"
group = r"("+str+r"\S+)"
rules = {"value": r"?P<value>"+value}

# FROM 指令的正则表达式模式
FROM_PATTERN = re.compile(
    r"^(?:(?P<from_option>--platform)=("+rules['value']+r")\s+)?"
    r"(?P<image>[^\s:@]+)"
    r"(?::(?P<tag>[^\s@]+))?"
    r"(?:@(?P<digest>[^\s]+))?"
    r"(?:\s+(?i:as)\s+(?P<name>[^\s]+))?",
    re.IGNORECASE
)

# RUN 指令的正则表达式模式
RUN_PATTERN = re.compile(
    r"^((?P<run_option>--mount|--network|--security)=("+rules['value']+r")\s+)?(?P<command>(.)+)",
    re.IGNORECASE
)

# LABEL/ENV 指令的正则表达式模式
LABEL_ENV_PATTERN = re.compile(
    r"^((?P<key>"+key+r")=(\S+))",
    re.IGNORECASE
)

# EXPOSE 指令的正则表达式模式
EXPOSE_PATTERN = re.compile(
    r"^(?P<port>"+port+r")",
    re.IGNORECASE
)

# COPY 指令的正则表达式模式
COPY_PATTERN = re.compile(
    r"^((?P<copy_flag>--from|--chown|--chmod|--link|--parents|--exclude)=("+rules['value']+r")\s+)?"
    r"(?P<src>"+src+r"\s+)+(?P<dest>"+dest+r")",
    re.IGNORECASE
)

# ADD 指令的正则表达式模式
ADD_PATTERN = re.compile(
    r"^((?P<add_flag>--checksum|--chown|--chmod|--keep-git-dir|--link=|--exclude)=("+rules['value']+r")\s+)?"
    r"(?P<src>"+src+r"\s+)+(?P<dest>"+dest+r")",
    re.IGNORECASE
)

# VOLUME 指令的正则表达式模式
VOLUME_PATTERN = re.compile(
    r"^((?P<json_path>\[.*?\])|(?P<path>\S+))",
    re.IGNORECASE
)

# USER 指令的正则表达式模式
USER_PATTERN = re.compile(
    r"^(?P<user>\S+)(:(?P<group>\S+))?",
    re.IGNORECASE
)

# HEALTHCHECK 指令的正则表达式模式
HEALTHCHECK_PATTERN = re.compile(
    r"^((?P<healthcheck_option>--interval|--timeout|--start-period|--start-interval|--retries)=("+rules['value']+r")(\s+)?)*"
    r"(CMD\s+(?P<command>[^\n]+))?",
    re.IGNORECASE
)


#parsed_tokens = []

# 函数：解析 Dockerfile 内容
def parse_dockerfile(dockerfile_content,parsed_tokens):
    lines = dockerfile_content.strip().splitlines()
    #print(lines)

    for line in lines:
        # 匹配指令和参数
        match = INSTRUCTION_PATTERN.match(line)
        #print(match)
        if match:
            instruction = match.group("instruction").upper()
            arguments = match.group("arguments").strip()

            # 根据指令进一步解析参数
            if instruction == "FROM":
                parsed_tokens.append((instruction, "FROM"))
                from_match = FROM_PATTERN.match(arguments)
                if from_match:
                    if from_match.group("from_option"):
                        parsed_tokens.append((from_match.group("from_option"), "FROM_OPTION"))
                        parsed_tokens.append(("=", "EQUALS"))
                        parsed_tokens.append((from_match.group("value"), "VALUE"))

                    parsed_tokens.append((from_match.group("image"), "IMAGE"))

                    if from_match.group("tag"):
                        parsed_tokens.append((":", "COLON"))
                        parsed_tokens.append((from_match.group("tag"), "TAG"))

                    if from_match.group("digest"):
                        parsed_tokens.append(("@", "AT"))
                        parsed_tokens.append((from_match.group("digest"), "DIGEST"))

                    if from_match.group("name"):
                        parsed_tokens.append(("AS", "AS"))
                        parsed_tokens.append((from_match.group("name"), "NAME"))

            elif instruction == "RUN":
                parsed_tokens.append((instruction, "RUN"))
                run_match = RUN_PATTERN.match(arguments)
                if run_match:
                    options = run_match.group("run_option")
                    command = run_match.group("command")

                    if options:
                        for option in options.split():
                            #print(option)
                            parsed_tokens.append((option, "RUN_OPTION"))
                            parsed_tokens.append(("=", "EQUALS"))
                            parsed_tokens.append((run_match.group("value"), "VALUE"))

                    if command.startswith("[") and command.endswith("]"):
                        parsed_tokens.append((command, "COMMAND"))
                        #parsed_tokens.append(("[", "LEFT_BRACKET"))
                        #if command.find(",")==-1:
                        #    parsed_tokens.append((command[1:-1], "COMMAND"))
                        #for item in command[1:-1].split(",")[:-1]:
                        #    parsed_tokens.append((item, "COMMAND"))
                        #    parsed_tokens.append((",", "COMMA"))
                        #parsed_tokens.append((command[1:-1].split(",")[-1], "COMMAND"))
                        #parsed_tokens.append(("]", "RIGHT_BRACKET"))
                    elif command.find("&&")!=-1 or command.find("||")!=-1 or command.find(";")!=-1:
                        while(command.find("&&")!=-1):
                            #print(command)
                            parsed_tokens.append((command[:command.find("&&")].strip(), "COMMAND"))
                            parsed_tokens.append(("&&", "AND"))
                            command = command[command.find("&&")+2:]
                            #print(command)
                        while(command.find("||")!=-1):
                            parsed_tokens.append((command[:command.find("||")].strip(), "COMMAND"))
                            parsed_tokens.append(("||", "AND"))
                            command = command[command.find("||")+2:]
                        while(command.find(";")!=-1):
                            parsed_tokens.append((command[:command.find(";")].strip(), "COMMAND"))
                            parsed_tokens.append((";", "AND"))
                            command = command[command.find(";")+2:]
                        parsed_tokens.append((command.strip(), "COMMAND"))
                    else:
                        parsed_tokens.append((arguments, "COMMAND"))
                #print(parsed_tokens)

            elif instruction in ["CMD", "ENTRYPOINT"]:
                parsed_tokens.append((instruction, instruction))
                if arguments.startswith("[") and arguments.endswith("]"):
                    #parsed_tokens.append(("[", "LEFT_BRACKET"))
                    parsed_tokens.append((arguments, "COMMAND"))
                    #parsed_tokens.append(("]", "RIGHT_BRACKET"))
                else:
                    parsed_tokens.append((arguments, "COMMAND"))

            elif instruction in ["LABEL", "ENV"]:
                parsed_tokens.append((instruction, instruction))

                if arguments.find("=") != -1:
                    #print(arguments)
                    for argument in arguments.split(" "):
                        #print(argument)
                        if argument.find("=")!=-1:
                            #label_env_match = LABEL_ENV_PATTERN.match(argument)
                            #if label_env_match:
                                #print(1)
                                #parsed_tokens.append((label_env_match.group("key"), "KEY"))
                            parsed_tokens.append((arguments.split("=")[0], "KEY"))
                            parsed_tokens.append(("=", "EQUALS"))
                            parsed_tokens.append((arguments.split("=")[1], "VALUE"))
                        else:
                            parsed_tokens.append((argument, "KEY"))
                else:
                    if len(arguments)>0:
                        parsed_tokens.append((arguments.split()[0], "KEY"))
                        try:
                            parsed_tokens.append((arguments.split()[1], "VALUE"))
                        except IndexError:
                            print("error")
                #print(parsed_tokens)

            elif instruction == "EXPOSE":
                parsed_tokens.append((instruction, "EXPOSE"))
                expose_match = EXPOSE_PATTERN.match(arguments)
                if expose_match:
                    if expose_match.group("port"):
                        parsed_tokens.append((expose_match.group("port"), "PORT"))

            elif instruction == "COPY":
                parsed_tokens.append((instruction, "COPY"))
                copy_match = COPY_PATTERN.match(arguments)

                if copy_match:
                    if copy_match.group("copy_flag"):
                        parsed_tokens.append((copy_match.group("copy_flag"), "COPY_FLAG"))
                        parsed_tokens.append(("=", "EQUALS"))
                        parsed_tokens.append((copy_match.group("value"), "VALUE"))

                try:
                    srcs = copy_match.group("src")
                    dest = copy_match.group("dest")
                except AttributeError:
                    parsed_tokens.append((arguments, "DESTINATION"))
                    srcs = 0
                    dest = 0
                if srcs:
                    for src in srcs.split():
                        #print(1)
                        parsed_tokens.append((src, "SOURCE"))
                if dest:
                    parsed_tokens.append((copy_match.group("dest"), "DESTINATION"))

            elif instruction == "ADD":
                parsed_tokens.append((instruction, "ADD"))
                add_match = ADD_PATTERN.match(arguments)

                if add_match:
                    if add_match.group("add_flag"):
                        parsed_tokens.append((add_match.group("add_flag"), "ADD_FLAG"))
                        parsed_tokens.append(("=", "EQUALS"))
                        parsed_tokens.append((add_match.group("value"), "VALUE"))

                try:
                    srcs = add_match.group("src")
                    dest = add_match.group("dest")
                except AttributeError:
                    parsed_tokens.append((arguments, "DESTINATION"))
                    srcs = 0
                    dest = 0
                    #continue
                #dest = add_match.group("dest")
                if srcs:
                    for src in srcs.split():
                        parsed_tokens.append((src, "SOURCE"))
                if dest:
                    parsed_tokens.append((add_match.group("dest"), "DESTINATION"))
                #print(parsed_tokens)

            elif instruction == "VOLUME":
                parsed_tokens.append((instruction, "VOLUME"))
                volume_match = VOLUME_PATTERN.match(arguments)

                if volume_match:
                    if volume_match.group("json_path"):
                        #parsed_tokens.append(("[", "LEFT_BRACKET"))
                        parsed_tokens.append((volume_match.group("json_path"), "PATH"))
                        #parsed_tokens.append(("]", "RIGHT_BRACKET"))
                    if volume_match.group("path"):
                        parsed_tokens.append((volume_match.group("path"), "PATH"))

            elif instruction == "USER":
                parsed_tokens.append((instruction, "USER"))
                user_match = USER_PATTERN.match(arguments)

                if user_match:
                    if user_match.group("user"):
                        parsed_tokens.append((user_match.group("user"), "USER_VALUE"))
                    if user_match.group("group"):
                        parsed_tokens.append((":", "COLON"))
                        parsed_tokens.append((user_match.group("group"), "GROUP"))

            elif instruction == "WORKDIR":
                parsed_tokens.append((instruction, "WORKDIR"))
                parsed_tokens.append((arguments, "PATH"))

            elif instruction == "ARG":
                parsed_tokens.append((instruction, "ARG"))
                if arguments.find("=")!=-1:
                    parsed_tokens.append((arguments.split("=")[0], "KEY"))
                    parsed_tokens.append(("=", "EQUALS"))
                    parsed_tokens.append((arguments.split("=")[1], "VALUE"))
                else:
                    parsed_tokens.append((arguments, "KEY"))

            elif instruction == "STOPSIGNAL":
                parsed_tokens.append((instruction, "STOPSIGNAL"))
                parsed_tokens.append((arguments, "VALUE"))

            elif instruction == "HEALTHCHECK":
                parsed_tokens.append((instruction, "HEALTHCHECK"))

                for argument in arguments.split():
                    #print(argument)
                    healthcheck_match = HEALTHCHECK_PATTERN.match(argument)
                    if healthcheck_match.group("healthcheck_option"):
                        #print("yes")
                        parsed_tokens.append((healthcheck_match.group("healthcheck_option"), "HEALTHCHECK_OPTION"))
                        parsed_tokens.append(("=", "EQUALS"))
                        parsed_tokens.append((healthcheck_match.group("value"), "VALUE"))


                    if healthcheck_match:
                        if healthcheck_match.group("command"):
                            parsed_tokens.append(("CMD", "CMD"))
                            parsed_tokens.append((healthcheck_match.group("command"), "COMMAND"))

            elif instruction == "SHELL":
                parsed_tokens.append((instruction, "SHELL"))
                parsed_tokens.append((arguments, "COMMAND"))

            elif instruction == "MAINTAINER":
                parsed_tokens.append((instruction, "MAINTAINER"))
                parsed_tokens.append((arguments, "VALUE"))

            elif instruction == "ONBUILD":
                parsed_tokens.append((instruction, "ONBUILD"))
                parse_dockerfile(arguments, parsed_tokens)
                #parsed_tokens.append((arguments, "VALUE"))

        elif line.startswith("#"):
            continue
        else:
            # 非指令行
            parsed_tokens.append((line.strip(), "UNKNOWN"))

    return parsed_tokens



if __name__ == '__main__':
    # repo = 'openfoodfacts-server'
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='1234', db='dockerfile_completion',
                           charset='utf8')
    cur = conn.cursor()
    conn_u = pymysql.connect(host='localhost', port=3306, user='root', passwd='1234', db='dockerfile_completion',
                             charset='utf8')
    cur_u = conn_u.cursor()
    mis_count = 0
    try:
        query = "select no,content_valid,token_ids from dockerfile_rmdup_3m where length(token_ids)<>length(token_ids_type)"
        #query = "select no,content_valid,token_ids from dockerfile_rmdup_3m where no=10083"
        cur.execute(query)
        data = cur.fetchall()
        for row in data:
            print(row[0])
            # getGHAWorkflows(fullname)
            #print("Dockerfile:")
            #print(row[1])
            # 解析 Dockerfile 内容
            try:
                parsed_tokens = parse_dockerfile(row[1],[])
                tag = 1
            except e:
            #    print(e)
                tag = 0
            #print("Tokens:")
            #print(row[2])
            #print(len([e for e in row[2][1:-1].split(',')]))
            #print(len(row[2]))

            # 打印解析结果
            #print("\nABSTRACTED:")
            if tag==1:
                types = []
                ids = []
                for token, token_type in parsed_tokens:
                    type_list = [token_type for _ in range(len(tokenizer.encode(token)))]
                    #print(f"{token}->{tokenizer.encode(token)}: {type_list}->{len(type_list)}")
                    types = types + type_list
                    ids = ids + tokenizer.encode(token)

                #print(len(types))
                if len(types)!=len([e for e in row[2][1:-1].split(',')]):
                    mis_count = mis_count + 1

                try:
                    query_u = "update dockerfile_rmdup_3m set token_type=\"%s\",token_ids_type=\"%s\" where no=%s" \
                              % (types,ids,row[0])
                    cur_u.execute(query_u)
                except pymysql.Error as e:
                    print("Mysql Error!", e);

    except pymysql.Error as e:
        print(e)
    conn.close()
    conn_u.close()

print("MIS:%s"%mis_count)
