import requests
import os
import sys
import json
import time

# 默认 Flask 服务地址
API_URL = "http://localhost:6001/api/upload_video"
VideoPath = "/workspace/data/input/test9.mp4"

def test_success_case():
    """测试成功情况 (200)"""
    print("=" * 50)
    print("🧪 测试用例 1: 成功处理 MP4 文件")
    print("=" * 50)
    
    video_path = VideoPath
    if not os.path.exists(video_path):
        print(f"❌ 测试文件不存在: {video_path}")
        return False

    files = {'file': open(video_path, 'rb')}
    data = {'max_seq_len': '10'}

    print(f"📤 上传视频: {video_path}")
    print(f"➡️  目标接口: {API_URL}")
    print(f"📊 参数: max_seq_len=10")

    try:
        start_time = time.time()
        response = requests.post(API_URL, files=files, data=data, timeout=1800)
        end_time = time.time()
        
        print(f"⏱️  处理时间: {end_time - start_time:.2f} 秒")
        print(f"📥 状态码: {response.status_code}")

        result = response.json()
        print("🧾 响应内容:")
        print(json.dumps(result, indent=4, ensure_ascii=False))

        if response.status_code == 200 and result.get("code") == 200:
            print("✅ 测试成功！")
            print(f"📹 处理后视频下载地址: {result['file_url']}")
            return True
        else:
            print("❌ 成功用例测试失败")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return False
    except ValueError:
        print("❌ 无法解析服务器返回的 JSON")
        print(f"响应内容: {response.text}")
        return False
    finally:
        files['file'].close()

def test_400_cases():
    """测试 400 客户端错误情况"""
    print("\n" + "=" * 50)
    print("🧪 测试用例 2: 客户端错误 (400)")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "无文件部分",
            "files": {},
            "data": {},
            "expected_message": "No file part"
        },
        {
            "name": "空文件名",
            "files": {'file': ('', b'', 'video/mp4')},
            "data": {},
            "expected_message": "No selected file"
        },
        {
            "name": "不支持的文件格式 (M4V)",
            "files": {'file': open('/workspace/data/input/test1.m4v', 'rb')} if os.path.exists('/workspace/data/input/test1.m4v') else None,
            "data": {},
            "expected_message": "Invalid file type, only MP4 is allowed"
        }
    ]

    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 子测试 {i}: {test_case['name']} ---")
        
        if test_case['files'].get('file') is None and test_case['name'] == "不支持的文件格式 (M4V)":
            print("⚠️  跳过测试 (M4V 文件不存在)")
            continue
            
        try:
            response = requests.post(API_URL, files=test_case['files'], data=test_case['data'], timeout=30)
            print(f"📥 状态码: {response.status_code}")
            
            result = response.json()
            print("🧾 响应内容:")
            print(json.dumps(result, indent=4, ensure_ascii=False))
            
            if (response.status_code == 400 and 
                result.get("code") == 400 and 
                test_case['expected_message'] in result.get("message", "")):
                print("✅ 400错误测试通过")
            else:
                print("❌ 400错误测试失败")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            all_passed = False
        except ValueError:
            print("❌ 无法解析服务器返回的 JSON")
            print(f"响应内容: {response.text}")
            all_passed = False
        finally:
            # 关闭文件句柄
            if test_case['files'].get('file') and hasattr(test_case['files']['file'], 'close'):
                test_case['files']['file'].close()
    
    return all_passed

def test_500_case():
    """测试 500 服务器错误情况（通过损坏的文件）"""
    print("\n" + "=" * 50)
    print("🧪 测试用例 3: 服务器错误 (500)")
    print("=" * 50)
    
    # 创建一个损坏的 MP4 文件来触发服务器错误
    corrupted_file_path = "/workspace/data/input/corrupted_test.mp4"
    
    # 创建损坏的MP4文件（只有文件头，没有有效内容）
    try:
        with open(corrupted_file_path, 'wb') as f:
            # 写入一些无效数据模拟损坏的MP4文件
            f.write(b'corrupted mp4 data that will cause server error')
        print(f"📁 创建损坏的测试文件: {corrupted_file_path}")
        
        files = {'file': open(corrupted_file_path, 'rb')}
        data = {'max_seq_len': '10'}
        
        print("📤 上传损坏的MP4文件...")
        response = requests.post(API_URL, files=files, data=data, timeout=30)
        print(f"📥 状态码: {response.status_code}")
        
        result = response.json()
        print("🧾 响应内容:")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        
        if response.status_code == 500 and result.get("code") == 500:
            print("✅ 500错误测试通过")
            # 清理测试文件
            os.remove(corrupted_file_path)
            return True
        else:
            print("❌ 500错误测试失败")
            # 清理测试文件
            os.remove(corrupted_file_path)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        # 清理测试文件
        if os.path.exists(corrupted_file_path):
            os.remove(corrupted_file_path)
        return False
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        # 清理测试文件
        if os.path.exists(corrupted_file_path):
            os.remove(corrupted_file_path)
        return False
    finally:
        if 'files' in locals() and files.get('file'):
            files['file'].close()

def test_download_endpoint():
    """测试文件下载端点"""
    print("\n" + "=" * 50)
    print("🧪 测试用例 4: 文件下载端点")
    print("=" * 50)
    
    # 先上传一个文件获取下载URL
    video_path = "/workspace/data/input/test1.mp4"
    if not os.path.exists(video_path):
        print("❌ 测试文件不存在，跳过下载测试")
        return False
        
    try:
        # 上传文件
        files = {'file': open(video_path, 'rb')}
        data = {'max_seq_len': '5'}
        response = requests.post(API_URL, files=files, data=data, timeout=1800)
        
        if response.status_code == 200:
            result = response.json()
            download_url = result['file_url']
            print(f"📥 获取下载地址: {download_url}")
            
            # 测试下载
            download_response = requests.get(download_url, timeout=30)
            print(f"📥 下载状态码: {download_response.status_code}")
            
            if download_response.status_code == 200:
                content_length = len(download_response.content)
                print(f"✅ 下载测试成功，文件大小: {content_length} 字节")
                return True
            else:
                print("❌ 下载测试失败")
                return False
        else:
            print("❌ 无法获取下载URL")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return False
    finally:
        files['file'].close()

def main():
    """主测试函数"""
    print("🚀 开始全面测试视频超分辨率 API")
    print(f"🎯 目标服务器: {API_URL}")
    
    test_results = []
    
    # 测试 200 成功情况
    test_results.append(("200 成功处理", test_success_case()))
    
    # 测试 400 客户端错误
    test_results.append(("400 客户端错误", test_400_cases()))
    
    # 测试 500 服务器错误
    test_results.append(("500 服务器错误", test_500_case()))
    
    # 测试下载端点
    # test_results.append(("文件下载端点", test_download_endpoint()))
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed_count = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed_count += 1
    
    total_count = len(test_results)
    print(f"\n🎯 测试结果: {passed_count}/{total_count} 通过")
    
    if passed_count == total_count:
        print("🎉 所有测试用例通过！")
        return 0
    else:
        print("💥 部分测试用例失败，请检查日志")
        return 1

if __name__ == "__main__":
    # 检查必要的测试文件
    required_files = [
        "/workspace/data/input/test1.mp4"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ 缺少必要的测试文件:")
        for f in missing_files:
            print(f"  - {f}")
        print("请确保测试文件存在后再运行测试")
        sys.exit(1)
    
    # 运行测试
    exit_code = main()
    sys.exit(exit_code)