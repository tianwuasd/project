"""Final direct-policy and log-selection regressions; stdlib control runtime."""
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import live_resources
import task_queries


class FinalIntegration(unittest.TestCase):
    def test_ram_capped_workers_reconsider_two_threads_without_reducing_workers(self):
        batch=64*2**20
        row=live_resources._rank_setting(23,23*batch,batch,16,0,0)
        self.assertEqual((row['train_workers'],row['worker_threads']),(5,2))
        self.assertLessEqual(row['main_threads']+row['train_worker_limit']*row['worker_threads'],23)
        self.assertLessEqual(row['estimated_ram_bytes'],23*batch)
        policy=live_resources.training_policy(dict(cpu_budget=191,memory_budget_bytes=256*2**30),8,batch)
        self.assertTrue(all((r['train_workers'],r['worker_threads'])==(16,1) for r in policy['rank_settings']))
        capacity=live_resources.training_policy(dict(cpu_budget=23,memory_budget_bytes=23*batch),1,batch)
        window=dict(steps=20,seconds=30,wait_fraction=.5)
        feedback=live_resources.feedback_policy(capacity,[window,window],capacity)['settings']
        self.assertLessEqual(feedback['total_cpu_slots'],23)
        self.assertLessEqual(feedback['estimated_ram_bytes'],23*batch)

    def test_model_progress_distinguishes_attempted_successful_and_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            def save(relative,value):
                p=work/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8')
            save('00_admin/model_selection.json',dict(folds=['f'],methods=['S1']))
            base='07_experiments/formal/f_B0_seed0/'
            save(base+'run.json',dict(status='running',planned_epochs=250))
            save(base+'progress.rank0.json',dict(epoch=0,planned_epochs=250,phase='train',batch=250,total_batches=250,
                updated=1,train_updates=dict(attempted=250,successful=249,amp_skipped=1)))
            output='\n'.join(task_queries.model_lines(work))
            self.assertIn('训练尝试250',output);self.assertIn('成功参数更新249',output);self.assertIn('AMP溢出跳过1',output)

    def test_nested_logs_have_unique_keys_and_selection_reads_only_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);area='08_results/evaluation/queue_logs'
            for name in ('worker0/worker.log','worker1/worker.log','flat.log'):
                p=work/area/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name,encoding='utf-8')
            with patch.object(task_queries,'context',return_value=({},work,{})):
                sources=task_queries.log_sources(work)
                self.assertEqual(len({key for key,_,_ in sources}),3)
                self.assertIn(area+':flat',[key for key,_,_ in sources])
                key=next(key for key,_,p in sources if p.parent.name=='worker0')
                output=task_queries.query('logs',work,work,log_source=key)
                self.assertIn('worker0/worker.log',output)
                self.assertNotIn('worker1',output)


if __name__=='__main__':unittest.main()
