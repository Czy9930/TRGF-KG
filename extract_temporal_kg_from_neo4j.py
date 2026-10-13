import os
import sys
import json
from collections import defaultdict, Counter

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "data",
    "temporal_kg"
)


RELATION2ID = {
    "disaster": 0,
    "data_source": 1,
    "HAS_OBSERVATION": 2,
    "NEXT_OBS": 3,
}

ID2RELATION = {
    v: k
    for k, v in RELATION2ID.items()
}


class TyphoonKGExporter:

    def __init__(self, uri, user, password):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )


        self.entity2id = {}
        self.id2entity = {}


        self.neo4j_to_entity = {}


        self.entity_metadata = []


        self.entity_time_mapping = []
        self.mapped_obs_entity_ids = set()


        self.triples = []


        self.triple_keys = set()


    def close(self):

        self.driver.close()


    @staticmethod
    def make_typhoon_id(tfbh, name):


        tfbh = str(tfbh).strip()
        name = str(name).strip()

        if (
            tfbh.startswith("20")
            and len(tfbh) == 6
        ):
            short_id = tfbh[2:]
        else:
            short_id = tfbh

        return f"{short_id}{name}"


    @staticmethod
    def safe_text(value, default="UNKNOWN"):

        if value is None:
            return default

        value = str(value).strip()

        if value == "":
            return default

        return value


    def add_entity(
        self,
        neo4j_id,
        entity_name,
        node_type,
        typhoon_id=None,
        extra=None
    ):


        if neo4j_id in self.neo4j_to_entity:

            return self.neo4j_to_entity[
                neo4j_id
            ]

        entity_name = str(entity_name)


        unique_name = entity_name

        if unique_name in self.entity2id:

            unique_name = (
                f"{entity_name}"
                f"__neo4j_{neo4j_id}"
            )

        entity_id = len(self.entity2id)

        self.entity2id[
            unique_name
        ] = entity_id

        self.id2entity[
            entity_id
        ] = unique_name

        self.neo4j_to_entity[
            neo4j_id
        ] = entity_id

        record = {
            "entity_id": entity_id,
            "entity_name": unique_name,
            "node_type": node_type,
            "neo4j_node_id": neo4j_id,
            "typhoon_id": typhoon_id
        }

        if extra is not None:

            record.update(extra)

        self.entity_metadata.append(
            record
        )

        return entity_id


    def add_triple(
        self,
        head_id,
        tail_id,
        relation,
        time_step,
        typhoon_id,
        head_type,
        tail_type
    ):

        relation_id = RELATION2ID[
            relation
        ]

        key = (
            head_id,
            tail_id,
            relation_id,
            int(time_step),
            str(typhoon_id)
        )

        if key in self.triple_keys:

            return

        self.triple_keys.add(key)

        self.triples.append({
            "head_id": head_id,
            "tail_id": tail_id,
            "relation_id": relation_id,
            "relation": relation,
            "time_step": int(time_step),
            "typhoon_id": str(typhoon_id),
            "head_type": head_type,
            "tail_type": tail_type
        })


    def extract_main_structure(self):

        print("\n" + "=" * 80)
        print("步骤1：提取 Category → Event → Source → OBS")
        print("=" * 80)

        query = """
        MATCH (category:Category)-[:disaster]->(event:Typhoon)
              -[:data_source]->(source:Source)
              -[:HAS_OBSERVATION]->(obs:OBS)

        WHERE obs.tfbh IS NOT NULL
          AND obs.name IS NOT NULL
          AND obs.observation_index IS NOT NULL

        RETURN
            id(category) AS category_node_id,
            labels(category) AS category_labels,
            properties(category) AS category_props,

            id(event) AS event_node_id,
            labels(event) AS event_labels,
            properties(event) AS event_props,

            id(source) AS source_node_id,
            labels(source) AS source_labels,
            properties(source) AS source_props,

            id(obs) AS obs_node_id,
            properties(obs) AS obs_props

        ORDER BY
            obs.tfbh,
            obs.observation_index
        """

        count = 0

        with self.driver.session() as session:

            result = session.run(query)

            for record in tqdm(
                result,
                desc="提取主结构"
            ):

                obs_props = dict(
                    record["obs_props"]
                )

                tfbh = self.safe_text(
                    obs_props.get("tfbh")
                )

                typhoon_name = self.safe_text(
                    obs_props.get("name")
                )

                obs_index = int(
                    obs_props.get(
                        "observation_index"
                    )
                )

                typhoon_id = (
                    self.make_typhoon_id(
                        tfbh,
                        typhoon_name
                    )
                )


                category_props = dict(
                    record["category_props"]
                )

                category_name = self.safe_text(
                    category_props.get(
                        "name",
                        category_props.get(
                            "value",
                            "台风"
                        )
                    ),
                    "台风"
                )

                category_entity_name = (
                    f"CATEGORY_{category_name}"
                )

                category_id = self.add_entity(
                    neo4j_id=
                        record[
                            "category_node_id"
                        ],
                    entity_name=
                        category_entity_name,
                    node_type="Category",
                    typhoon_id=None
                )


                event_entity_name = (
                    f"EVENT_{typhoon_id}"
                )

                event_id = self.add_entity(
                    neo4j_id=
                        record[
                            "event_node_id"
                        ],
                    entity_name=
                        event_entity_name,
                    node_type="TyphoonEvent",
                    typhoon_id=
                        typhoon_id,
                    extra={
                        "tfbh": tfbh,
                        "typhoon_name":
                            typhoon_name
                    }
                )


                source_entity_name = (
                    f"SOURCE_{typhoon_id}_"
                    f"{record['source_node_id']}"
                )

                source_id = self.add_entity(
                    neo4j_id=
                        record[
                            "source_node_id"
                        ],
                    entity_name=
                        source_entity_name,
                    node_type="Source",
                    typhoon_id=
                        typhoon_id
                )


                obs_entity_name = (
                    f"{typhoon_id}"
                    f"_OBS_{obs_index}"
                )

                obs_id = self.add_entity(
                    neo4j_id=
                        record[
                            "obs_node_id"
                        ],
                    entity_name=
                        obs_entity_name,
                    node_type="Observation",
                    typhoon_id=
                        typhoon_id,
                    extra={
                        "tfbh": tfbh,
                        "time_step":
                            obs_index
                    }
                )


                if obs_id not in self.mapped_obs_entity_ids:
                    self.mapped_obs_entity_ids.add(obs_id)

                    self.entity_time_mapping.append({
                        "entity_id":
                            obs_id,

                        "entity_name":
                            obs_entity_name,

                        "typhoon_id":
                            typhoon_id,

                        "tfbh":
                            tfbh,

                        "time_step":
                            obs_index,

                        "lat":
                            obs_props.get(
                                "lat"
                            ),

                        "lng":
                            obs_props.get(
                                "lng"
                            ),

                        "speed":
                            obs_props.get(
                                "speed"
                            ),

                        "pressure":
                            obs_props.get(
                                "pressure"
                            ),

                        "move_speed":
                            obs_props.get(
                                "move_speed"
                            ),

                        "move_dir":
                            obs_props.get(
                                "move_dir"
                            )
                    })


                self.add_triple(
                    category_id,
                    event_id,
                    "disaster",
                    -1,
                    typhoon_id,
                    "Category",
                    "TyphoonEvent"
                )


                self.add_triple(
                    event_id,
                    source_id,
                    "data_source",
                    -1,
                    typhoon_id,
                    "TyphoonEvent",
                    "Source"
                )


                self.add_triple(
                    source_id,
                    obs_id,
                    "HAS_OBSERVATION",
                    obs_index,
                    typhoon_id,
                    "Source",
                    "Observation"
                )

                count += 1

        print(
            f"✓ 主结构 OBS 记录数: {count}"
        )
        if count == 0:
            raise ValueError("Neo4j 中未找到完整的 Category→Typhoon→Source→OBS 路径")


    def extract_next_obs(self):

        print("\n" + "=" * 80)
        print("步骤2：提取 NEXT_OBS 时序关系")
        print("=" * 80)

        query = """
        MATCH (o1:OBS)-[:NEXT_OBS]->(o2:OBS)

        WHERE o1.tfbh IS NOT NULL
          AND o2.tfbh IS NOT NULL
          AND o1.name IS NOT NULL
          AND o2.name IS NOT NULL
          AND o1.observation_index IS NOT NULL
          AND o2.observation_index IS NOT NULL
          AND o1.tfbh = o2.tfbh

        RETURN
            id(o1) AS o1_node_id,
            properties(o1) AS o1_props,

            id(o2) AS o2_node_id,
            properties(o2) AS o2_props

        ORDER BY
            o1.tfbh,
            o1.observation_index
        """

        count = 0
        skipped = 0

        with self.driver.session() as session:

            result = session.run(query)

            for record in tqdm(
                result,
                desc="提取NEXT_OBS"
            ):

                o1_props = dict(
                    record["o1_props"]
                )

                o2_props = dict(
                    record["o2_props"]
                )

                tfbh = self.safe_text(
                    o1_props.get("tfbh")
                )

                name = self.safe_text(
                    o1_props.get("name")
                )

                typhoon_id = (
                    self.make_typhoon_id(
                        tfbh,
                        name
                    )
                )

                t1 = int(
                    o1_props[
                        "observation_index"
                    ]
                )

                t2 = int(
                    o2_props[
                        "observation_index"
                    ]
                )

                o1_neo4j_id = record[
                    "o1_node_id"
                ]

                o2_neo4j_id = record[
                    "o2_node_id"
                ]


                if (
                    o1_neo4j_id
                    not in
                    self.neo4j_to_entity
                ):
                    skipped += 1
                    continue

                if (
                    o2_neo4j_id
                    not in
                    self.neo4j_to_entity
                ):
                    skipped += 1
                    continue

                o1_id = (
                    self.neo4j_to_entity[
                        o1_neo4j_id
                    ]
                )

                o2_id = (
                    self.neo4j_to_entity[
                        o2_neo4j_id
                    ]
                )


                if t2 <= t1:
                    raise ValueError(
                        f"NEXT_OBS 时间异常 {typhoon_id}: {t1} -> {t2}"
                    )

                self.add_triple(
                    o1_id,
                    o2_id,
                    "NEXT_OBS",


                    t1,

                    typhoon_id,
                    "Observation",
                    "Observation"
                )

                count += 1

        print(
            f"✓ NEXT_OBS: {count}"
        )

        print(
            f"  因主结构不存在而跳过: "
            f"{skipped}"
        )
        if skipped:
            raise ValueError(f"{skipped} 条 NEXT_OBS 边的端点不在主结构中")


    def validate(self):

        print("\n" + "=" * 80)
        print("步骤3：知识图谱完整性检查")
        print("=" * 80)

        df_entity = pd.DataFrame(
            self.entity_metadata
        )

        df_obs = pd.DataFrame(
            self.entity_time_mapping
        )

        df_triples = pd.DataFrame(
            self.triples
        )

        print("\n实体类型：")

        print(
            df_entity[
                "node_type"
            ].value_counts()
        )

        print("\n关系类型：")

        print(
            df_triples[
                "relation"
            ].value_counts()
        )

        print(
            f"\n台风数量: "
            f"{df_obs['typhoon_id'].nunique()}"
        )

        print(
            f"OBS数量: "
            f"{len(df_obs)}"
        )


        duplicate_obs = (
            df_obs.groupby(
                [
                    "typhoon_id",
                    "time_step"
                ]
            )
            .size()
        )

        if (
            duplicate_obs > 1
        ).any():

            raise ValueError(
                "发现同一台风同一time_step"
                "存在多个OBS实体"
            )


        obs_rel = (
            df_triples[
                df_triples[
                    "relation"
                ]
                ==
                "HAS_OBSERVATION"
            ]
        )

        triple_typhoons = set(
            obs_rel[
                "typhoon_id"
            ]
        )

        obs_typhoons = set(
            df_obs[
                "typhoon_id"
            ]
        )

        missing = (
            obs_typhoons
            -
            triple_typhoons
        )

        if missing:

            raise ValueError(
                f"以下台风缺少"
                f"HAS_OBSERVATION: "
                f"{list(missing)[:10]}"
            )


        id_to_typhoon = {
            int(row["entity_id"]):
                row["typhoon_id"]
            for _, row
            in df_entity.iterrows()
        }

        next_rows = (
            df_triples[
                df_triples[
                    "relation"
                ]
                ==
                "NEXT_OBS"
            ]
        )

        for _, row in (
            next_rows.iterrows()
        ):

            h_ty = id_to_typhoon.get(
                int(row["head_id"])
            )

            t_ty = id_to_typhoon.get(
                int(row["tail_id"])
            )

            if h_ty != t_ty:

                raise ValueError(
                    "发现跨台风NEXT_OBS："
                    f"{h_ty} -> {t_ty}"
                )

        metadata_by_id = {
            int(row["entity_id"]): row
            for row in self.entity_metadata
        }
        obs_by_id = {
            int(row["entity_id"]): row
            for row in self.entity_time_mapping
        }
        expected_types = {
            "disaster": ("Category", "TyphoonEvent"),
            "data_source": ("TyphoonEvent", "Source"),
            "HAS_OBSERVATION": ("Source", "Observation"),
            "NEXT_OBS": ("Observation", "Observation"),
        }
        has_obs_counts = Counter()
        for triple in self.triples:
            head = triple["head_id"]
            tail = triple["tail_id"]
            relation = triple["relation"]
            if head not in metadata_by_id or tail not in metadata_by_id:
                raise ValueError("三元组引用了不存在的实体")
            if triple["relation_id"] != RELATION2ID[relation]:
                raise ValueError("三元组的 relation_id 与 relation 不一致")
            actual_types = (
                metadata_by_id[head]["node_type"],
                metadata_by_id[tail]["node_type"],
            )
            if actual_types != expected_types[relation]:
                raise ValueError(
                    f"{relation} 的端点类型错误: {actual_types}"
                )
            for entity_id in (head, tail):
                owner = metadata_by_id[entity_id]["typhoon_id"]
                if owner is not None and owner != triple["typhoon_id"]:
                    raise ValueError(
                        f"{relation} 的 typhoon_id 与实体归属不一致"
                    )
            if relation in ("disaster", "data_source"):
                if triple["time_step"] != -1:
                    raise ValueError(f"{relation} 的 time_step 必须为 -1")
            elif relation == "HAS_OBSERVATION":
                has_obs_counts[tail] += 1
                if triple["time_step"] != obs_by_id[tail]["time_step"]:
                    raise ValueError("HAS_OBSERVATION 时间步与 OBS 不一致")
            elif relation == "NEXT_OBS":
                if triple["time_step"] != obs_by_id[head]["time_step"]:
                    raise ValueError("NEXT_OBS 时间步必须是 head OBS 时间步")
                if obs_by_id[tail]["time_step"] <= obs_by_id[head]["time_step"]:
                    raise ValueError("NEXT_OBS 的 tail 时间必须晚于 head")
        if set(has_obs_counts) != set(obs_by_id):
            raise ValueError("有 OBS 缺少 HAS_OBSERVATION 边")
        if any(count != 1 for count in has_obs_counts.values()):
            raise ValueError("有 OBS 拥有多条 HAS_OBSERVATION 边")

        manifest_path = os.path.join(
            SCRIPT_DIR, "data", "input", "typhoon_split_manifest.csv"
        )
        if os.path.exists(manifest_path):
            manifest = pd.read_csv(
                manifest_path, encoding="utf-8-sig",
                dtype={"typhoon_id": str}
            )
            missing_storms = set(manifest["typhoon_id"]) - obs_typhoons
            if missing_storms:
                raise ValueError(
                    f"预测数据集中的 {len(missing_storms)} 个台风不在导出的 KG 中"
                )
            print(
                f"✓ 预测数据集的 {len(manifest)} 个台风均在新 KG 中"
            )

        print(
            "\n✓ 图结构检查通过"
        )


    def save(self, output_dir):

        print("\n" + "=" * 80)
        print("步骤4：保存知识图谱")
        print("=" * 80)

        os.makedirs(
            output_dir,
            exist_ok=True
        )


        entity_file = os.path.join(
            output_dir,
            "temporal_entities.txt"
        )

        with open(
            entity_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                f"{len(self.entity2id)}\n"
            )

            for eid in sorted(
                self.id2entity.keys()
            ):

                f.write(
                    f"{self.id2entity[eid]}"
                    f"\t{eid}\n"
                )


        relation_file = os.path.join(
            output_dir,
            "temporal_relations.txt"
        )

        with open(
            relation_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                f"{len(RELATION2ID)}\n"
            )

            for rel, rid in (
                RELATION2ID.items()
            ):

                f.write(
                    f"{rel}\t{rid}\n"
                )


        triple_file = os.path.join(
            output_dir,
            "temporal_triples.txt"
        )

        with open(
            triple_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                f"{len(self.triples)}\n"
            )

            for x in self.triples:

                f.write(
                    f"{x['head_id']}\t"
                    f"{x['tail_id']}\t"
                    f"{x['relation_id']}\t"
                    f"{x['time_step']}\n"
                )


        df_triples = pd.DataFrame(
            self.triples
        )

        full_triple_file = os.path.join(
            output_dir,
            "temporal_triples_full.csv"
        )

        df_triples.to_csv(
            full_triple_file,
            index=False,
            encoding="utf-8-sig"
        )


        df_mapping = pd.DataFrame(
            self.entity_time_mapping
        )

        mapping_file = os.path.join(
            output_dir,
            "entity_time_mapping.csv"
        )

        df_mapping.to_csv(
            mapping_file,
            index=False,
            encoding="utf-8-sig"
        )


        df_entity = pd.DataFrame(
            self.entity_metadata
        )

        metadata_file = os.path.join(
            output_dir,
            "entity_metadata.csv"
        )

        df_entity.to_csv(
            metadata_file,
            index=False,
            encoding="utf-8-sig"
        )


        temporal_index = {}

        for typhoon_id, group in (
            df_mapping.groupby(
                "typhoon_id"
            )
        ):

            group = group.sort_values(
                "time_step"
            )

            temporal_index[
                typhoon_id
            ] = {
                "entity_ids":
                    group[
                        "entity_id"
                    ]
                    .astype(int)
                    .tolist(),

                "time_steps":
                    group[
                        "time_step"
                    ]
                    .astype(int)
                    .tolist(),

                "num_steps":
                    len(group),

                "min_time":
                    int(
                        group[
                            "time_step"
                        ].min()
                    ),

                "max_time":
                    int(
                        group[
                            "time_step"
                        ].max()
                    )
            }

        index_file = os.path.join(
            output_dir,
            "typhoon_temporal_index.json"
        )

        with open(
            index_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                temporal_index,
                f,
                ensure_ascii=False,
                indent=2
            )


        relation_counts = (
            Counter(
                x["relation"]
                for x in self.triples
            )
        )

        node_counts = (
            Counter(
                x["node_type"]
                for x in
                self.entity_metadata
            )
        )

        stats = {
            "num_entities":
                len(self.entity2id),

            "num_relations":
                len(RELATION2ID),

            "num_triples":
                len(self.triples),

            "num_typhoons":
                int(
                    df_mapping[
                        "typhoon_id"
                    ].nunique()
                ),

            "num_observations":
                len(df_mapping),

            "node_type_counts":
                dict(node_counts),

            "relation_counts":
                dict(
                    relation_counts
                )
        }

        stats_file = os.path.join(
            output_dir,
            "kg_statistics.json"
        )

        with open(
            stats_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                stats,
                f,
                ensure_ascii=False,
                indent=2
            )

        print("\n✓ 保存完成")
        print(f"  {entity_file}")
        print(f"  {relation_file}")
        print(f"  {triple_file}")
        print(f"  {full_triple_file}")
        print(f"  {mapping_file}")
        print(f"  {metadata_file}")
        print(f"  {index_file}")
        print(f"  {stats_file}")


    def export(self, output_dir):

        print("=" * 80)
        print("台风路径预测任务专用知识图谱导出")
        print("=" * 80)

        self.extract_main_structure()

        self.extract_next_obs()

        self.validate()

        self.save(
            output_dir
        )

        print("\n" + "=" * 80)
        print("导出完成")
        print("=" * 80)


def main():
    if not NEO4J_PASSWORD:
        raise ValueError('Set the NEO4J_PASSWORD environment variable.')

    print(
        f"Neo4j URI: "
        f"{NEO4J_URI}"
    )

    print(
        f"输出目录: "
        f"{OUTPUT_DIR}"
    )

    exporter = None

    try:

        exporter = TyphoonKGExporter(
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD
        )

        exporter.export(
            OUTPUT_DIR
        )

    except Exception as e:

        print(
            f"\n导出失败: {e}"
        )

        import traceback

        traceback.print_exc()
        raise

    finally:

        if exporter is not None:

            exporter.close()


if __name__ == "__main__":

    main()
