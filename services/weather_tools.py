import requests
import json
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
from config.settings import Settings

# 获取日志记录器 ：从Python的logging模块中获取一个logger对象
# __name__ 是当前模块的名称（这里是 services.weather_tools ）
logger = logging.getLogger(__name__)

class WeatherService:
    """天气查询服务类"""

    def __init__(self):
        self.settings = Settings()
        self.api_key = self.settings.WEATHER_API_KEY
        self.weather_url = self.settings.WEATHER_API_URL
        self.city_url = self.settings.WEATHER_CITY_URL

        # 城市代码缓存
        self.city_cache = {}

    def get_city_code(self, city_name: str) -> Optional[str]:
        """获取城市代码"""
        """
        - 作用 ：根据城市名称获取对应的天气数据接口城市代码
        - 界面元素 ：城市名称输入框
        - 返回 ：城市代码（成功）或 None（失败）
        """
        try:
            # 检查缓存
            if city_name in self.city_cache:
                return self.city_cache[city_name]

            # 构建请求URL
            url = f"{self.city_url}"
            params = {
                "keywords": city_name,
                "subdistrict": 0,
                "key": self.api_key,
                "extensions": "base"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("status") == "1" and data.get("districts"):
                # 获取第一个匹配的城市
                districts = data["districts"]
                if districts and len(districts) > 0:
                    city_code = districts[0].get("adcode")
                    if city_code:
                        # 缓存结果
                        self.city_cache[city_name] = city_code
                        logger.info(f"获取城市代码成功: {city_name} -> {city_code}")
                        return city_code

            logger.warning(f"未找到城市: {city_name}")
            return None

        except requests.RequestException as e:
            logger.error(f"获取城市代码失败: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"获取城市代码出错: {str(e)}")
            return None

    def get_current_weather(self, city_name: str) -> str:
        """获取当前天气"""
        try:
            city_code = self.get_city_code(city_name)
            if not city_code:
                return f"抱歉，无法找到城市 '{city_name}' 的信息。请检查城市名称是否正确。"

            # 构建请求URL
            params = {
                "city": city_code,
                "key": self.api_key,
                "extensions": "base"  # base=实况天气
            }

            response = requests.get(self.weather_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("status") == "1" and data.get("lives"):
                weather_info = data["lives"][0]

                # 格式化天气信息
                result = self._format_current_weather(weather_info, city_name)
                logger.info(f"获取当前天气成功: {city_name}")
                return result
            else:
                error_msg = data.get("info", "未知错误")
                logger.warning(f"获取当前天气失败: {error_msg}")
                return f"获取天气信息失败: {error_msg}"

        except requests.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"获取当前天气失败: {error_msg}")
            return f"获取天气信息失败，请稍后重试。"
        except Exception as e:
            error_msg = f"获取当前天气出错: {str(e)}"
            logger.error(error_msg)
            return f"获取天气信息时发生错误: {str(e)}"

    def get_weather_forecast(self, city_name: str, days: int = 3) -> str:
        """获取天气预报"""
        """
        - 作用 ：根据城市名称获取未来几天的天气预报
        - 界面元素 ：城市名称输入框 + 预报天数选择器
        - 返回 ：天气预报字符串（成功）或错误信息（失败）
        """
        try:
            if days < 1 or days > 7:
                return "预报天数必须在1-7天之间。"

            city_code = self.get_city_code(city_name)
            if not city_code:
                return f"抱歉，无法找到城市 '{city_name}' 的信息。请检查城市名称是否正确。"

            # 构建请求URL
            params = {
                "city": city_code,
                "key": self.api_key,
                "extensions": "all"  # all=预报天气
            }

            response = requests.get(self.weather_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("status") == "1" and data.get("forecasts"):
                forecast_info = data["forecasts"][0]

                # 格式化预报信息
                result = self._format_weather_forecast(forecast_info, city_name, days)
                logger.info(f"获取天气预报成功: {city_name}, 天数: {days}")
                return result
            else:
                error_msg = data.get("info", "未知错误")
                logger.warning(f"获取天气预报失败: {error_msg}")
                return f"获取天气预报失败: {error_msg}"

        except requests.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"获取天气预报失败: {error_msg}")
            return f"获取天气预报失败，请稍后重试。"
        except Exception as e:
            error_msg = f"获取天气预报出错: {str(e)}"
            logger.error(error_msg)
            return f"获取天气预报时发生错误: {str(e)}"    

    def _format_current_weather(self, weather_data: Dict[str, Any], city_name: str) -> str:
        """格式化当前天气信息"""
        """
        - 作用 ：将天气数据格式化为易读的字符串
        - 界面元素 ：城市名称 + 当前天气数据
        - 返回 ：格式化后的天气字符串
        """
        try:
            province = weather_data.get("province", "")
            city = weather_data.get("city", city_name)
            weather = weather_data.get("weather", "")
            temperature = weather_data.get("temperature", "")
            winddirection = weather_data.get("winddirection", "")
            windpower = weather_data.get("windpower", "")
            humidity = weather_data.get("humidity", "")
            reporttime = weather_data.get("reporttime", "")

            # 构建格式化输出
            result = f"🏙️ **{province} {city}** 当前天气\n\n"
            result += f"🌤️ **天气状况**: {weather}\n"
            result += f"🌡️ **气温**: {temperature}°C\n"
            result += f"💨 **风向风力**: {winddirection} {windpower}\n"
            result += f"💧 **湿度**: {humidity}%\n"
            result += f"📅 **发布时间**: {reporttime}\n"

            # 添加天气建议
            result += "\n💡 **温馨提示**:\n"

            if temperature and temperature.isdigit():
                temp = int(temperature)
                if temp < 10:
                    result += "• 天气较冷，请注意保暖。\n"
                elif temp > 30:
                    result += "• 天气较热，请注意防暑。\n"
                else:
                    result += "• 天气舒适，适合外出。\n"

            if humidity and humidity.isdigit():
                hum = int(humidity)
                if hum > 80:
                    result += "• 湿度较高，注意防潮。\n"
                elif hum < 30:
                    result += "• 湿度较低，注意补水。\n"

            return result

        except Exception as e:
            logger.error(f"格式化当前天气信息失败: {str(e)}")
            return f"天气数据格式化失败: {str(e)}"

    def _format_weather_forecast(self, forecast_data: Dict[str, Any], city_name: str, days: int) -> str:
        """格式化天气预报信息"""
        """
        - 作用 ：将预报数据格式化为易读的字符串
        - 界面元素 ：城市名称 + 预报天数
        - 返回 ：格式化后的预报字符串
        """
        try:
            province = forecast_data.get("province", "")
            city = forecast_data.get("city", city_name)
            reporttime = forecast_data.get("reporttime", "")
            casts = forecast_data.get("casts", [])

            result = f"🏙️ **{province} {city}** 未来{days}天天气预报\n\n"
            result += f"📅 **发布时间**: {reporttime}\n\n"

            # 只显示指定天数
            for i, cast in enumerate(casts[:days]):
                date = cast.get("date", "")
                week = cast.get("week", "")
                dayweather = cast.get("dayweather", "")
                nightweather = cast.get("nightweather", "")
                daytemp = cast.get("daytemp", "")
                nighttemp = cast.get("nighttemp", "")
                daywind = cast.get("daywind", "")
                nightwind = cast.get("nightwind", "")
                daypower = cast.get("daypower", "")
                nightpower = cast.get("nightpower", "")

                result += f"📅 **{date}** ({week})\n"
                result += f"🌤️ **天气**: 白天{dayweather}，夜间{nightweather}\n"
                result += f"🌡️ **温度**: 白天{daytemp}°C，夜间{nighttemp}°C\n"
                result += f"💨 **风力**: 白天{daywind}{daypower}，夜间{nightwind}{nightpower}\n"
                
                if i < days - 1:  # 不是最后一天就加分割线
                    result += "\n" + "─" * 30 + "\n\n"

            return result

        except Exception as e:
            logger.error(f"格式化天气预报信息失败: {str(e)}")
            return f"预报数据格式化失败: {str(e)}"


if __name__ == "__main__":
    """
    天气服务测试代码
    运行方法：python services/weather_tools.py
    """
    import logging
    
    # 配置日志显示
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("🌤️ 天气服务测试开始...")
    print("=" * 50)
    
    # 创建天气服务实例
    weather_service = WeatherService()
    
    # 测试1：获取城市代码
    print("\n📍 测试1：获取城市代码")
    print("-" * 30)
    test_cities = ["北京", "上海"]
    
    for city in test_cities:
        city_code = weather_service.get_city_code(city)
        if city_code:
            print(f"✅ {city}: {city_code}")
        else:
            print(f"❌ {city}: 未找到城市代码")
    
    # 测试2：获取当前天气
    print("\n🌡️ 测试2：获取当前天气")
    print("-" * 30)
    
    for city in test_cities[:3]:  # 只测试前3个城市
        print(f"\n🌍 {city}当前天气：")
        weather_info = weather_service.get_current_weather(city)
        print(weather_info)
        print("-" * 30)
    
    # 测试3：获取天气预报
    print("\n📅 测试3：获取天气预报")
    print("-" * 30)
    
    for city in test_cities[:2]:  # 只测试前2个城市
        for days in [1, 3, 5]:
            print(f"\n🌈 {city}未来{days}天预报：")
            forecast_info = weather_service.get_weather_forecast(city, days)
            print(forecast_info)
            print("-" * 30)
    
    # 测试4：错误处理
    print("\n⚠️ 测试4：错误处理")
    print("-" * 30)
    
    # 测试不存在的城市
    fake_city = "不存在的城市123"
    result = weather_service.get_current_weather(fake_city)
    print(f"查询不存在的城市 '{fake_city}':")
    print(result)
    
    # 测试无效的预报天数
    result = weather_service.get_weather_forecast("北京", 0)
    print(f"\n预报天数为0：")
    print(result)
    
    result = weather_service.get_weather_forecast("北京", 10)
    print(f"\n预报天数为10：")
    print(result)
    
    print("\n" + "=" * 50)
    print("🎉 天气服务测试完成！")
    print("\n💡 测试结果说明：")
    print("• ✅ 表示功能正常")
    print("• ❌ 表示有错误或找不到数据")
    print("• 如果看到天气信息，说明API调用成功")
    print("• 如果看到错误提示，说明错误处理机制工作正常")